""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
np.seterr(divide='ignore', invalid='ignore') #permet d'ignorer les division par 0

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factorpong for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(2,H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in iter(model.items()) } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in iter(model.items()) } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
# pour l'instant on reste sur du 80 80, pour rester simple
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
	#
    discounted_r = np.zeros_like(r)
    discounted_r[r.size-1] = r[r.size-1]
    for i in reversed(range(0, r.size-1)):
        discounted_r[i] = r[i] + gamma*discounted_r[i+1]
    return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity, fonction d'activation
  logp1 = np.dot(model['W2'][0], h)
  logp2 = np.dot(model['W2'][1], h)
  p1 = sigmoid(logp1)
  p2 = sigmoid(logp2)
  return p1, p2, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.array((np.dot(eph.T, epdlogp[0]).ravel(),np.dot(eph.T, epdlogp[1]).ravel()))
  dh = np.array((np.outer(epdlogp[0], model['W2'][0]),np.outer(epdlogp[1], model['W2'][1])), ndmin=2)
  dh[0][eph <= 0] = 0 # backpro prelu, fonction d'activation
  dh[1][eph <= 0] = 0
  dW1 = np.array((np.dot(dh[0].T, epx),np.dot(dh[1].T, epx))) 
  return {'W1':dW1, 'W2':dW2}

env = gym.make("SpaceInvaders-v0")
observation = env.reset()
prev_x = None # utilisé pour capté la difference des images, et donc detecter un "mouvement" des objets
xs,hs,dlogps,drs = [],[],[[],[]],[]
running_reward = None
reward_sum = 0
episode_number = 0
epdlogp = [[],[]]
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  probaMove, probaFire, h = policy_forward(x)

  moveLeft = True if np.random.uniform() < probaMove else False # roll the dice!
  fire = True if np.random.uniform() < probaFire else False
  action = 0
  if(moveLeft):
      action = 4 if fire else 2
  else:
      action = 5 if fire else 3
      
  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if moveLeft else 0 # a "fake label" 
  dlogps[0].append(y - probaMove) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused) stock des loss
  y = 1 if fire else 0 # a "fake label" 
  dlogps[1].append(y - probaFire)
  
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp[0] = np.vstack(dlogps[0])
    epdlogp[1] = np.vstack(dlogps[1])
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[[],[]],[] # reset array memory
    

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewardsto be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    deviation = np.std(discounted_epr)
    not_all_zeros = np.any(deviation)
    if(not_all_zeros):
        discounted_epr /= deviation

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    epdlogp = [[],[]]
    grad_buffer['W1'] += grad['W1'][0]
    grad_buffer['W1'] += grad['W1'][1]
    grad_buffer['W2'][0] += grad['W2'][0]
    grad_buffer['W2'][1] += grad['W2'][1]
    
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in iter(model.items()):
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
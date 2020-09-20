""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factorpong for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # ueresume from previous checkpoint?
render = False
actions=[2,3,1]

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  model['W3'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W4'] = np.random.randn(H) / np.sqrt(H)
  model['W5'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W6'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in iter(model.items()) } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in iter(model.items()) } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro is cropping, and preprocessing the image before difference """
  #les valeurs du background sont noir, donc ils sont deja a 0
  I = I[35:195] #crop 84x84
  I = I[::2,::2,0] # applique 0 tout les deux pixels. permet de voir le déplacement si of fait cur_x - prev_x car la difference sera bien plus grande.
  I[I != 0] = 1 #tout les elements utile passe a 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  discounted_r[r.size-1] = r[r.size-1]
  for i in reversed(range(0, r.size-1)):
    discounted_r[i] = r[i] + gamma*discounted_r[i+1]
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)

  h2 = np.dot(model['W3'], x)
  h2[h2<0] = 0 # ReLU nonlinearity
  logp2 = np.dot(model['W4'], h2)
  p2 = sigmoid(logp2)

  h3 = np.dot(model['W5'], x)
  h3[h3<0] = 0 # ReLU nonlinearity
  logp3 = np.dot(model['W6'], h3)
  p3 = sigmoid(logp3)

  return p, h, p2, h2, p3, h3

def policy_backward(eph, epdlogp, eph2, epdlogp2, eph3, epdlogp3):
  """ backward pass. (eph is array of intermediate hidden states) """
  # premier reseau
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)

  #Second reseau deplacement
  dW4 = np.dot(eph2.T, epdlogp2).ravel()
  dh2 = np.outer(epdlogp2, model['W4'])
  dh2[eph2 <= 0] = 0 # backpro prelu
  dW3 = np.dot(dh2.T, epx)

  #troisieme reseau tire
  dW6 = np.dot(eph3.T, epdlogp3).ravel()
  dh3 = np.outer(epdlogp3, model['W6'])
  dh3[eph3 <= 0] = 0 # backpro prelu
  dW5 = np.dot(dh3.T, epx)
  return {'W1':dW1, 'W2':dW2, 'W3':dW3, 'W4':dW4, 'W5':dW5, 'W6':dW6}


def getAction(P):
  sumP = 0
  for k in P:
    sumP += k
  rng = np.random.uniform()
  i = 0
  sumK = 0
  if(sumP == 0) :
      print("SumP == 0")
  for k in P:
    sumK += k
    if rng <= sumK/sumP:
      return actions[i]
    i = i + 1
  return 0



env = gym.make("SpaceInvaders-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
hs2,hs3,dlogps2,dlogps3 = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h, aprob2, h2, aprob3, h3 = policy_forward(x)
  P = [aprob,aprob2,aprob3]
  action = getAction(P)

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  hs2.append(h2) # hidden state2
  hs3.append(h3) # hidden state3

  y = 1 if action == 1 else 0 # a "fake label"
  dlogps3.append(y - aprob3) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  y = 1 if action == 3 else 0 # a "fake label"
  dlogps2.append(y - aprob2) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    eph2 = np.vstack(hs2)
    epdlogp2 = np.vstack(dlogps2)
    eph3 = np.vstack(hs3)
    epdlogp3 = np.vstack(dlogps3)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory
    hs2,hs3,dlogps2,dlogps3 = [],[],[],[] # reset array memory


    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards pongto be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    deviation = np.std(discounted_epr)
    not_all_zeros = np.any(deviation)
    if(not_all_zeros):
        discounted_epr /= deviation

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    epdlogp2 *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    epdlogp3 *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)


    grad = policy_backward(eph, epdlogp,eph2,epdlogp2,eph3,epdlogp3)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

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

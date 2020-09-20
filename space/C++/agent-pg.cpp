#include <iostream>
#include <ale_interface.hpp>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <math.h>
#include <numeric>

#ifdef __USE_SDL
#include <SDL.h>
#endif

using namespace std;

const int H = 200;
const int batch_size = 10;
const float learning_rate = 1e-4;
const float gammaconst = 0.99;
const float decay_rate = 0.99;
const bool resume = false;
const bool render = false;
float D = 80 * 80;

int sigmoid(float x){
  float resf;
  resf = 1.0 / (1.0 + exp(x-1));
  int res = (int)resf;
  return res;
}

vector<float> prepro(vector<vector<vector<float>>> I){
  vector<float> res(D);
  for (int i=35; i<195; i+2){
    for (int y=0; y<160; y+2){
      for (int z=0; z<3; z++){
        if (I[i][y][z] == 144){
          res.push_back(0);
        }
        if (I[i][y][z] == 109) {
          res.push_back(0);
        }
        if (I[i][y][z] != 0) {
          res.push_back(1);
        }
      }
    }
  }
  return res;
}

vector<float> discount_rewards(vector<float> r){
  vector<float> discounted_r(r.size(),0.0);
  float running_add=0.0;
  for (int i=r.size()-1; i>-1;i--){
    if (r[i] != 0.0){
      running_add = 0.0;
    }
    running_add = running_add * gammaconst * r[i];
    discounted_r[i]=running_add;
  }
  return discounted_r;
}

//faire multiplication
tuple<int,vector<float>> policy_forward(vector<float> x){
  float h;
  float logp;
  int p = sigmoid(logp);
  tuple<int,vector<float>> res(p,h);
  return res;
}

//faire multiplication
map<string,vector<vector<float>>> policy_backward(vector<vector<float>> eph,vector<float> epdlogp){
  map<string,vector<vector<float>>> res;
  vector<float> hvec(H);
  vector<vector<float>> dw2(1,hvec);
  vector<vector<float>> dh;
  vector<vector<float>> dw1(D,hvec);
  res["W1"] = dw1;
  res["W2"] = dw2;
  return res;
}


int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
    return 1;
  }

  ale::ALEInterface ale;

  // ale.setInt("random_seed", 115);
  // ale.setFloat("repeat_action_probability", 0.25);

// #ifdef __USE_SDL
  // ale.setBool("display_screen", true);
  // ale.setBool("sound", true);
// #endif

  // Load the ROM file. (Also resets the system for new settings to
  // take effect.)
  ale.loadROM(argv[1]);

  // Get the vector of legal actions
  ale::ActionVect legal_actions = ale.getLegalActionSet();


  map<string,vector<vector<float>>> model;
  vector<float> hvec(H);
  vector<float> hvec0(H,0.0);
  vector<vector<float>> w2(1,hvec0);
  vector<vector<float>> w1(D,hvec0);
  model["W1"] = w1;
  model["W2"] = w2;

  map<string,vector<vector<float>>> grad_buffer(model);
  map<string,vector<vector<float>>> rmsprop_cache(model);

  vector<float> drs,cur_x,prev_x,x;
  vector<float> dlogps;
  vector<vector<float>> xs,hs;

  float running_reward,reward_sum;
  int episode_number;

  cur_x = prepro();//observation)

  vector<float> difference;
  set_difference(
      cur_x.begin(), cur_x.end(),
      prev_x.begin(), prev_x.end(),
      std::back_inserter( difference )
  );

  if (prev_x.empty() == false ){
    x = difference;
  }
  else {
    vector<float> x(D,0.0);
  }

  prev_x = cur_x;

  tuple<int,vector<float>> aprobetH = policy_forward(x);
  int aprob = get<0>(aprobetH);
  vector<float> h = get<1>(aprobetH);

  int action;
  vector<float> vecOfRandomNums(x.size());
  generate(vecOfRandomNums.begin(), vecOfRandomNums.end(), []() {
	return rand() % 2;
});
  if (2 < aprob ){
    action = 2;
  }
  else {
    action = 3;
  }

  xs.push_back(x);
  hs.push_back(h);


  int y;
  if (action==2) {
    y = 2;
  }
  else {
    y = 0;
  }

  dlogps.push_back((float)(y-aprob));

  //  observation, reward, done, info = env.step(action)
  // il faut encore récupérer l'observation
  bool done;
  float reward;  // reward_t reward;
  int info;
  reward_sum += (float)reward;

  drs.push_back((float)reward);

  if (done){
    episode_number++;
    vector<vector<float>> epx(xs);
    vector<vector<float>> eph(hs);
    vector<float> epdlogp(dlogps);
    vector<float> epr(drs);
    xs.clear();
    hs.clear();
    epdlogp.clear();
    drs.clear();

    vector<float> discounted_epr = discount_rewards(epr);

    float sum =accumulate(discounted_epr.begin(), discounted_epr.end(), 0.0);
    float mean = sum / discounted_epr.size();

    vector<float> diff(discounted_epr.size());
    transform(discounted_epr.begin(), discounted_epr.end(), diff.begin(),// modifier, imparallelisable
               bind2nd(minus<float>(), mean));
    float sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = sqrt(sq_sum / discounted_epr.size());

    for (int i=0; i<discounted_epr.size(); i++){
      discounted_epr.at(i) -= mean;
      discounted_epr.at(i) /= stdev;
    }

    transform(epdlogp.begin(),epdlogp.end(),discounted_epr.begin(),epdlogp.begin(),multiplies<float>()); // modifier, imparallelisable

    map<string,vector<vector<float>>> grad(policy_backward(eph,epdlogp));
    transform(grad_buffer["W1"][0].begin(),grad_buffer["W1"][0].end(),grad["W1"][0].begin(),grad_buffer["W1"][0].begin(),plus<float>());// modifier, imparallelisable
    transform(grad_buffer["W2"][0].begin(),grad_buffer["W2"][0].end(),grad["W2"][0].begin(),grad_buffer["W2"][0].begin(),plus<float>());// modifier, imparallelisable

    if (episode_number % batch_size == 10 ){
      vector<vector<float>> g1(grad_buffer["W1"]);
      for (int i = 0; rmsprop_cache["W1"][0].size(); i++){
        rmsprop_cache["W1"][0][i] = decay_rate * rmsprop_cache["W1"][0][i] + (1.0 - decay_rate) * pow(g1[0][i],2);
        model["W1"][0][i] += learning_rate * g1[0][i] / (sqrt(rmsprop_cache["W1"][0][i]) + 1e-5);
      }

      vector<vector<float>> g2(grad_buffer["W2"]);
      for (int i = 0; rmsprop_cache["W2"][0].size(); i++){
        rmsprop_cache["W2"][0][i] = decay_rate * rmsprop_cache["W2"][0][i] + (1.0 - decay_rate) * pow(g2[0][i],2);
        model["W2"][0][i] += learning_rate * g2[0][i] / (sqrt(rmsprop_cache["W2"][0][i]) + 1e-5);

      }
      grad_buffer["W1"].clear();
      grad_buffer["W2"].clear();

    }

    if (isnan(running_reward)){
      running_reward = reward_sum;
    }
    else {
      running_reward = running_reward * 0.99 + reward_sum * 0.01;
    }
  }

  // Pour 10 épisodes , en cours de réalisations
  for (int episode = 0; episode < 10; episode++) {
    float totalReward = 0;
    while (!ale.game_over()) {
      ale::Action a = legal_actions[rand() % legal_actions.size()]; //action aléatoire pour le moment
      float reward = ale.act(a);
      totalReward += reward;
    }
    cout << "Episode " << episode << " ended with score: " << totalReward
         << endl;
    ale.reset_game();
  }

  return 0;
}

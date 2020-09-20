//
// Created by calgary on 05/05/2020.
//
#include <iostream>
#include <ale_interface.hpp>
#include <ale_interface.cpp>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <math.h>
#include <numeric>
#include "moon_lib/Moon.h"
#include "moon_lib/Moon.cpp"

#ifdef __USE_SDL
#include <SDL.h>
#endif

int D = 80*80;
float gam = 0.99;
int H = 200;
int batch_size = 10;
int learning_rate = 1e-4;
int decay_rate = 0.99;
bool resume = false;
bool render = false;

/*grad_buffer = { k : np.zeros_like(v) for k,v in iter(model.items()) } # update buffers that add up gradients over a batch
        rmsprop_cache = { k : np.zeros_like(v) for k,v in iter(model.items()) } # rmsprop memory*/



float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}


mat::Matrix prepro(std::vector<unsigned char> I){
    mat::Matrix res(1,D);

    std::vector<float> prepro_colors;
    int height = 210;
    int width = 160;
    int pixsize = 3;
    int pos = 0;

    for (int i = 35; i < 195; i ++){
        if (i % 2 == 0){
            for (int j = 0; j < width; j ++){
                if ( j % 2 == 0){
                    for (int k = 0; k < 1; k ++){
                        pos = j + width*i + width*height*k;
                        if ((float)I.at(pos) != 0){
                            prepro_colors.push_back(1.0);
                        }
                        else {
                            prepro_colors.push_back((float)I.at(pos));
                        }
                    }
                }
            }
        }
    }
    for(int i =0; i<prepro_colors.size();i++){
        res.getArray()[i][0] = prepro_colors.at(i);
    }
    return res;
}

mat::Matrix discouted_rewards(mat::Matrix r){
    mat::Matrix dis_r = mat::zeros_like(r);
    dis_r.getArray()[0][r.getLength() -1 ] = r.getArray()[0][r.getLength() - 1];
    for(int i = r.getLength() -1 ; i != -1; i--){
        dis_r.getArray()[0][i] = r.getArray()[0][i] + gam* dis_r.getArray()[0][i+1];
    }
    return dis_r;
}

int policy_forward(mat::Matrix x, mat::Matrix W1, mat::Matrix W2, float* h, float* p1, float* p2){
    *h = mat::dotf(W1,x);
    if(*h < 0){
        *h = 0;
    }
    float logp1 = mat::dotf(W2,*h,0);
    float logp2 = mat::dotf(W2,*h,1);
    *p1 = sigmoid(logp1);
    *p2 = sigmoid(logp2);
    return 1;
}

int policy_backward(mat::Matrix eph, mat::Matrix epdlogp, mat::Matrix W1, mat::Matrix W2,mat::Matrix epx, std::vector<mat::Matrix> grad_buffer){
    float **epd1array;
    epd1array = new float *[0];
    epd1array[0] = epdlogp.getArray()[0];

    float **epd2array;
    epd2array = new float *[0];
    epd2array[0] = epdlogp.getArray()[1];

    mat::Matrix epd1 = mat::Matrix(epd1array, epdlogp.getLength(), 1);
    mat::Matrix epd2 = mat::Matrix(epd2array, epdlogp.getLength(), 1);
    mat::Matrix tmp1 = mat::dot(eph, epd1);
    tmp1.ravel();
    mat::Matrix tmp2 = mat::dot(eph,epd2);
    tmp2.ravel();
    mat::Matrix tmp3 = mat::outer(epdlogp,W2,0,0);
    tmp3.ravel();
    mat::Matrix tmp4 = mat::outer(epdlogp,W2,1,1);
    tmp4.ravel();
    mat::Matrix DW2 = mat::vstack(tmp1, tmp2);
    //definition du +

    grad_buffer.insert(grad_buffer.end(),mat::vstack(tmp1, tmp2));  //W2 à la place 1
    mat::Matrix dh = mat::vstack(tmp3, tmp4);

    for (int i = 0; i < dh.getHeight(); i ++){
        for(int j = 0; j < dh.getLength(); j++){
            if(dh.getArray()[i][j] < 0){
                dh.getArray()[i][j] = 0;
            }
        }
    }

    float **dh1array;
    dh1array = new float *[0];
    dh1array[0] = dh.getArray()[0];

    float **dh2array;
    dh2array = new float *[0];
    dh2array[0] = dh.getArray()[1];

    mat::Matrix dh1 = mat::Matrix(dh1array, dh.getLength(), 1);
    mat::Matrix dh2 = mat::Matrix(dh2array, dh.getLength(), 1);
    dh1.T();
    dh2.T();
    grad_buffer.insert(grad_buffer.end(),mat::vstack(mat::dot(dh1,epx),mat::dot(dh2,epx))); //W1 à la place 2

    mat::Matrix DW1 = mat::vstack(mat::dot(dh1,epx),mat::dot(dh2,epx));
    //definition du +


    grad_buffer.clear();
    grad_buffer.insert(grad_buffer.end(),DW1);
    grad_buffer.insert(grad_buffer.end(),DW2);
    return 1;
}


int main(int argc, char** argv){
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    int numbers = 300;
    mat::Matrix W1 = mat::Matrix(D,H);
    mat::Matrix W2 = mat::Matrix(H,2);
    ale::ALEInterface aleInterface;
    //aleInterface.setBool("display_screen",true);
    aleInterface.loadROM(argv[1]);
    std::vector<mat::Matrix> xs;
    std::vector<mat::Matrix> grad_buffer;
    mat::Matrix W1LIKE = mat::zeros_like(W1);
    mat::Matrix W2LIKE = mat::zeros_like(W2);
    grad_buffer.insert(grad_buffer.end(),W1LIKE);
    grad_buffer.insert(grad_buffer.end(),W2LIKE);
    std::vector<float> hs;
    std::vector<unsigned char> a;
    bool initialized = false;
    mat::Matrix x = mat::Matrix(D,1);
    std::vector<float> dlogs1;
    int reward_sum = 0;
    std::vector<float> dlogs2;
    std::vector<ale::reward_t> drs;
    int nbr_ep = 0;
    while(!aleInterface.game_over()){
        numbers--;

        aleInterface.getScreenRGB(a);
        mat::Matrix cur_x= prepro(a);
        mat::Matrix prev_x = mat::Matrix(D,1);
        cur_x.ravel();

        if(!initialized){
            initialized = true;
            x = mat::zeros_like(prev_x);
            //delete n
        }else{
            for(int i=0; i<D ; i++){
                x.getArray()[0][i] = cur_x.getArray()[0][i] - prev_x.getArray()[0][i];
            }
        }

        prev_x = cur_x;

        float probaMove = 0.0;
        float probaFire = 0.0;
        float h = 0.0;

        policy_forward(x,W1,W2,&h,&probaMove,&probaFire);
        bool moveLeft = false;
        bool fire = false;
        if(mat::randomU(0,1) < probaMove){
            moveLeft = true;
        }
        if(mat::randomU(0,1) < probaFire){
            fire = true;
        }
        int action = 0;
        if(moveLeft){
            if(fire){
                action = 4;
            }else{
                action = 2;
            }
        }else{
            if(fire){
                action = 5;
            }else{
                action = 3;
            }
        }

        xs.insert(xs.end(),x);
        hs.insert(hs.end(),h);

        if(moveLeft){
            dlogs1.insert(dlogs1.end(),(1 - probaMove));
        }else{
            dlogs1.insert(dlogs1.end(),(0 - probaMove));
        }
        if(fire){
            dlogs2.insert(dlogs2.end(),(1 - probaFire));
        }else{
            dlogs2.insert(dlogs2.end(),(0 - probaFire));
        }

        ale::Action acti = aleInterface.getMinimalActionSet().at(action);
        ale::reward_t reward = aleInterface.act(acti);

        reward_sum += reward;

        drs.insert(drs.end(),reward);

        if(aleInterface.game_over()){
            nbr_ep++;
            mat::Matrix epx = mat::Matrix(xs.at(0).getArray(),xs.at(0).getLength(),xs.at(0).getHeight());
            for(int i = 1; i < xs.size(); i++){
                std::cout <<" commence " << std::endl;
                epx.bstack(xs.at(i));
            }
            std::cout <<" fini " << std::endl;
            mat::Matrix eph = mat::Matrix((int)hs.size(),1);
            for(int i =0; i < hs.size(); i++){
                eph.getArray()[0][i] = hs.at(i);
            }
            eph.T();

            mat::Matrix epdlogp = mat::Matrix(dlogs1.size(),2);
            for(int i = 0; i< dlogs1.size();i++){
                epdlogp.getArray()[0][i]=dlogs2.at(i);
                epdlogp.getArray()[1][i]=dlogs1.at(i);
            }
            epdlogp.T();
            mat::Matrix epr = mat::Matrix(drs.size(),1);
            for(int i = 0; i< drs.size();i++){
                epdlogp.getArray()[0][i]=drs.at(i);
            }
            epr.T();

            xs.clear();
            hs.clear();
            dlogs2.clear();
            dlogs1.clear();
            drs.clear();

            mat::Matrix discounted_epr = discouted_rewards(epr);

            float mean = mat::mean(discounted_epr);

            for(int i = 0; i< discounted_epr.getHeight();i++){
                for(int j = 0; j< discounted_epr.getLength();j++){
                    discounted_epr.getArray()[i][j] = discounted_epr.getArray()[i][j] - mean;
                }
            }

            float deviation = 0.0;
            for(int i = 0; i< discounted_epr.getHeight();i++){
                for(int j = 0; j< discounted_epr.getLength();j++){
                    deviation = deviation + std::pow(discounted_epr.getArray()[i][j] - mean, 2);
                }
            }
            float Sdeviation = sqrt(deviation/(discounted_epr.getLength()*discounted_epr.getHeight()));

            if(Sdeviation != 0){
                for(int i = 0; i< discounted_epr.getHeight();i++){
                    for(int j = 0; j< discounted_epr.getLength();j++){
                        discounted_epr.getArray()[i][j] = discounted_epr.getArray()[i][j]/Sdeviation;
                    }
                }
            }
            mat::Matrix Repdlogp = mat::multiply(epdlogp,discounted_epr);
            policy_backward(eph,Repdlogp,W1,W2,epx,grad_buffer);
        }

    }

    return 1;
}

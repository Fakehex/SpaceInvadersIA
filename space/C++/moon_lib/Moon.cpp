//
// Created by calgary on 05/05/2020.
//
#include "Moon.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <omp.h>

mat::Matrix::Matrix(int length, int height) {
    _length = length;
    _height = height;

    float **t;
    t = new float *[height];
    for (int i = 0; i < height; ++i) {
        t[i] = new float[length];
    }
    srand((unsigned int) time(nullptr));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < length; j++) {
            t[i][j] = ((float) rand()) / (float) RAND_MAX;
        }
    }
    _array = t;
}

mat::Matrix::Matrix(float** array, int length, int height) {
    _length = length;
    _height = height;
    _array = array;
}

int mat::Matrix::getLength() {
    return _length;
}

int mat::Matrix::bstack(mat::Matrix m){
    if(_length == m.getLength()){
        float **myarray;
        myarray = new float *[_height+m.getHeight()];
        for (int i = 0; i < _height+m.getHeight(); ++i) {
            myarray[i] = new float[_length];
        }
        std::copy(&_array[0][0], &_array[0][0]+((_height+1)*m.getLength())+1,&myarray[0][0]);
        std::copy(&m.getArray()[0][0], &m.getArray()[0][0]+((m.getHeight()+1)*m.getLength())+1,&myarray[_height][0]);
        _height = _height + m.getHeight();
        delete _array;
        _array = myarray;
        return 1;
    }else{
        std::cout << "Matrices de longueur différentes (" << _length << "|" << m.getLength() << ")" << std::endl;
        exit(101);
    }
}

void Swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

int mat::Matrix::getHeight() {
    return _height;
}

int mat::Matrix::T() {
    float **myarray;

    myarray = new float *[_length];

    for (int i = 0; i < _length; ++i) {
        myarray[i] = new float[_height];
    }
    for (int i = 0; i < _length; ++i) {
        for (int j = 0; j < _height; ++j) {
            myarray[i][j] = _array[j][i];
        }
    }

    Swap(_height,_length);

    delete _array;
    _array = myarray;
    return 1;
}



int mat::Matrix::ravel() {
    float **myarray;
    myarray = new float *[0];
    myarray[0] = new float[_height*_length];

    int z = 0;
    for (int i = 0; i < _height; ++i) {
        for (int j = 0; j < _length; ++j){
            myarray[0][z] = _array[i][j];
            z++;
        }
    }

    _length = _length*_height;
    _height = 1;

    delete _array;
    _array = myarray;
    return 1;
}

float **mat::Matrix::getArray() {
    return _array;
}

int mat::Matrix::print() {
    for(int i = 0; i<_height; i++){
        for(int j = 0; j<_length; j++){
            std::cout << "|" << _array[i][j];
        }
        std::cout << "|"  << std::endl;
    }
    return 1;
}

mat::Matrix mat::vstack(mat::Matrix m1) {
    if(m1.getHeight() == 1){
        float** myarray;
        myarray = new float*[1];
        myarray[0] = new float[m1.getLength()];

        for(int j = 0; j < m1.getLength(); j++){
            myarray[0][j] = m1.getArray()[0][j];
        }
        int n = m1.getLength();
        int m = m1.getHeight();
        Matrix m2 = Matrix(myarray,n,m);
        m2.T();
        return m2;
    }
    return m1;
}

mat::Matrix mat::outer(mat::Matrix m1, mat::Matrix m2, int choosenm1, int choosenm2) {
    float **myarray;
    myarray = new float *[m1.getLength()];
    for (int i = 0; i < m1.getLength(); ++i) {
        myarray[i] = new float[m2.getLength()];
    }
    for(int i = 0; i < m1.getLength(); i++){
        for(int j = 0; j < m2.getLength(); j++){
            myarray[i][j] = m1.getArray()[choosenm1][i] * m2.getArray()[choosenm2][j];
        }
    }
    int n = m1.getLength();
    int m = m2.getLength();
    Matrix m3 = Matrix(myarray,m,n);
    return m3;
}

mat::Matrix mat::vstack(mat::Matrix m1, mat::Matrix m2) {
    if(m1.getLength() == m2.getLength()){
        float **myarray;
        myarray = new float *[m1.getHeight()+m2.getHeight()];
        for (int i = 0; i < m1.getHeight()+m2.getHeight(); ++i) {
            myarray[i] = new float[m1.getLength()];
        }
        for(int i = 0; i < m1.getHeight(); i++){
            for(int j = 0; j < m1.getLength(); j++){
                myarray[i][j] = m1.getArray()[i][j];
            }
        }
        for(int i = m1.getHeight(); i < m1.getHeight()+m2.getHeight(); ++i){
            for(int j = 0; j < m2.getLength(); j++){
                myarray[i][j] = m2.getArray()[i-m1.getHeight()][j];
            }
        }
        int n = m1.getLength();
        int m = m1.getHeight() + m2.getHeight();
        Matrix m3 = Matrix(myarray,n,m);
        return m3;
    }else{
        std::cout << "Matrices de longueur différentes (" << m1.getLength() << "|" << m2.getLength() << ")" << std::endl;
        exit(101);
    }
}

mat::Matrix mat::zeros_like(mat::Matrix m) {
    float **myarray;
    myarray = new float *[m.getHeight()];
    for (int i = 0; i < m.getHeight(); ++i) {
        myarray[i] = new float[m.getLength()];
    }

    for(int i = 0; i < m.getHeight();i++){
        for(int j = 0; j < m.getLength(); j++){
            myarray[i][j] = 0;
        }
    }
    Matrix m1 = Matrix(myarray,m.getLength(),m.getHeight());
    return m1;
}

mat::Matrix mat::sqrt(mat::Matrix m) {
    float **myarray;
    myarray = new float *[m.getHeight()];
    for (int i = 0; i < m.getHeight(); ++i) {
        myarray[i] = new float[m.getLength()];
    }

    for(int i = 0; i < m.getHeight();i++){
        for(int j = 0; j < m.getLength(); j++){
            myarray[i][j] = std::sqrt(m.getArray()[i][j]);
        }
    }
    Matrix m1 = Matrix(myarray,m.getLength(),m.getHeight());
    return m1;
}


mat::Matrix mat::dot(mat::Matrix m1, mat::Matrix m2) {
    if(m1.getHeight() > 1 && m2.getHeight() > 1) {
        if (m1.getHeight() == m2.getLength()) {
            float **myarray;
            myarray = new float *[m2.getHeight()];
            for (int i = 0; i < m2.getHeight(); ++i) {
                myarray[i] = new float[m1.getLength()];
            }
            for (int i = 0; i < m2.getHeight(); i++) {
                for (int j = 0; j < m1.getLength(); j++) {
                    int z = 0;
                    for (int k = 0; k < m1.getHeight(); ++k) {
                        z = z + m1.getArray()[k][j] * m2.getArray()[i][k];
                    }
                    myarray[i][j] = z;
                }
            }
            Matrix m3 = Matrix(myarray, m1.getLength(), m2.getHeight());
            return m3;
        }
    }
        if(m1.getLength() == m2.getHeight()) {
            float **myarray;
            myarray = new float *[m1.getHeight()];
            for (int i = 0; i < m1.getHeight(); ++i) {
                myarray[i] = new float[m2.getLength()];
            }
            for(int i = 0; i < m1.getHeight();i++){
                for(int j = 0; j < m2.getLength();j++){
                    int z = 0;
                    for (int k = 0; k < m2.getHeight(); ++k) {
                        z = z + m2.getArray()[k][j]* m1.getArray()[i][k];
                    }
                    myarray[i][j] = z;
                }
            }
            Matrix m3 = Matrix(myarray,m2.getLength(),m1.getHeight());
            return m3;
        }

    std::cout << "Impossible de multiplier les matrices" << std::endl;
    exit(102);
}

float mat::randomU(float low, float high) {
    srand((unsigned int) time(nullptr));
    return (((float) rand()) / (float) RAND_MAX)*high;
}

float mat::dotf(Matrix m1, Matrix m2) {
    if(m1.getHeight() == 1 && m2.getHeight() == 1){
        if(m1.getLength() != m2.getLength()){
            std::cout << "Matrices de longueur différentes (" << m1.getLength() << "|" << m2.getLength() << ")" << std::endl;
            exit(101);
        }
        float a = 0.0;
        for (int i = 0; i < m1.getLength(); i++) {
            a = a + m1.getArray()[0][i] * m2.getArray()[0][i];
        }
        return a;
    }
    if (m1.getHeight() > 1 && m2.getHeight() == 1 && m1.getLength()==m2.getLength()) {
        float a = 0.0;
        for (int i = 0; i < m1.getLength(); i++) {
            a = a + m1.getArray()[m1.getHeight() - 1][i] * m2.getArray()[0][i];
        }
        return a;
    }
    if (m1.getHeight() == 1 && m2.getHeight() > 1 && m1.getLength()==m2.getLength()) {
        float a = 0.0;
        for (int i = 0; i < m2.getLength(); i++) {
            a = a + m2.getArray()[m2.getHeight() - 1][i] * m1.getArray()[0][i];
        }
        return a;
    }
    std::cout << "Impossible" << std::endl;
    exit(103);
}

float mat::dotf(mat::Matrix m1, float f) {
    float a = 0;
    for(int i = 0; i < m1.getLength(); i++){
        a = a + m1.getArray()[m1.getHeight() - 1][i] * f;
    }
    return a;
}

mat::Matrix mat::exp(mat::Matrix m) {
    float **t;
    t = new float *[m.getHeight()];
    for (int i = 0; i < m.getHeight(); ++i) {
        t[i] = new float[m.getLength()];
    }
    for(int i = 0; i < m.getHeight(); i++){
        for(int j = 0; j < m.getLength(); j++){
            t[i][j] = std::exp(m.getArray()[i][j]);
        }
    }

    mat::Matrix matrix = mat::Matrix(t,m.getLength(),m.getHeight());
    return matrix;
}

float mat::dotf(mat::Matrix m1, float f, int choosen) {
    float a = 0;
    for(int i = 0; i < m1.getLength(); i++){
        a = a + m1.getArray()[choosen][i] * f;
    }
    return a;
}

float mat::mean(mat::Matrix m){
    float a = 0;
    for(int i =0; i < m.getHeight() ; i++){
        for(int j =0; j < m.getLength() ; j++){
            a = a + m.getArray()[i][j];
        }
    }
    a = a/(m.getLength()*m.getHeight());
    return a;
}

mat::Matrix mat::multiply(mat::Matrix m1, mat::Matrix m2){
    if(m2.getLength() == m1.getHeight()){
        float **t;
        t = new float *[m1.getHeight()];
        for (int i = 0; i < m1.getHeight(); ++i) {
            t[i] = new float[m1.getLength()];
        }
        for(int i = 0; i < m1.getHeight(); i++){
            for(int j = 0; j < m1.getLength();j++){
                t[i][j] = m2.getArray()[0][i];
            }
        }
        mat::Matrix m = mat::Matrix(t,m1.getLength(),m1.getHeight());
        return m;
    }
    if(m1.getLength() == m2.getHeight()){
        float **t;
        t = new float *[m2.getHeight()];
        for (int i = 0; i < m2.getHeight(); ++i) {
            t[i] = new float[m2.getLength()];
        }
        for(int i = 0; i < m2.getHeight(); i++){
            for(int j = 0; j < m2.getLength();j++){
                t[i][j] = m1.getArray()[0][i];
            }
        }
        mat::Matrix m = mat::Matrix(t,m2.getLength(),m2.getHeight());
        return m;
    }

    std::cout << "Impossible de multiply, ne concorde pas" << std::endl;
    exit(104);
}



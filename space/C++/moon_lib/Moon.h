//
// Created by calgary on 05/05/2020.
//

#ifndef INVADERS_MOON_H
#define INVADERS_MOON_H

namespace mat {

    class Matrix
    {
    private:
        int _length;
        int _height;
        float** _array;
    public:
        /**
         * Crée une Matrice aléatoire de taille _length x _height
         *
         * @param _length
         * @param _height
         */
        Matrix(int _length, int _height);
        /**
         * Crée une matrice avec les valeurs du tableau de taille length x height
         *
         * @param _array Tableau de la matrice de taille _length x _height
         * @param _length
         * @param _height
         */
        Matrix(float **_array, int _length,int _height);
        /**
         * @return _length le longueur de la matrice
         */
        int getLength();
        /**
         * @return _height la hauteur de la matrice
         */
        int getHeight();
        /**
         * @return float** le tableau de la matrice
         */
        float** getArray();

        int bstack(Matrix m);

        /**
         * Calcule la transposée
         * @return 1
         */
        int T();
        /**
         * Ecrase la matrice en une unique ligne
         * @return 1
         */
        int ravel();

        /**
         * Affiche la matrice
         *
         * @return 1
         */
        int print();

    };

    /**
     * Multiplications de matrices
     *
     * @param m1 une matrice
     * @param m2 une seconde matrice
     * @return une nouvelle matrice résultat
     */
    Matrix dot(Matrix m1, Matrix m2);

    /**
     * Multiplications d'un vecteur et d'une matrice
     *
     * @param m1 une matrice
     * @param m2 une seconde matrice
     * @return un float
     */
    float dotf(Matrix m1, Matrix m2);

    /**
     * Multiplications d'un float et d'une matrice
     *
     * @param m1 une matrice
     * @param f un float
     * @return un float
     */
    float dotf(Matrix m1, float f);

    /**
     * Multiplications d'un float et d'un tableau
     *
     * @param m1 une matrice
     * @param f un float
     * @param la ligne que l'on veux prendre
     * @return un float
     */
    float dotf(mat::Matrix m1, float f, int choosen);

    /**
     * Calcule le produit de vecteurs uniquement
     *
     * @param m1 matrice 1,D
     * @param m2 matrice 1,N
     * @return une nouvelle matrice résultat
     */
    Matrix outer(Matrix m1, Matrix m2, int choosenm1, int choosenm2);

    /**
     * Stack la matrice si possible
     *
     * @param m1 matrice H,L
     * @return une nouvelle matrice stackée si elle a été modifiées
     */
    Matrix vstack(Matrix m1);

    /**
     * Stack deux matrices
     *
     * @param m1 matrice H,L
     * @param m2 matrice N,L
     * @return une nouvelle matrice stackée
     */
    Matrix vstack(Matrix m1, Matrix m2);

    /**
     * Retourne une matrice de taille égale avec uniquement des 0
     *
     * @param matrice H,L
     * @return une nouvelle matrice avec des 0
     */
    Matrix zeros_like(Matrix m);

    /**
     * Retourne une matrice de taille égale mais avec les racines des valeurs
     *
     * @param m matrice H,L
     * @return une nouvelle matrice avec des valeurs au carré
     */
    Matrix sqrt(Matrix m);

    /**
     * Génère float entre low et high
     *
     * @param low borne inf
     * @param high boren sup
     * @return un float entre low et high
     */
    float randomU(float low, float high);

    mat::Matrix exp(mat::Matrix m);

    float mean(mat::Matrix m);

    mat::Matrix multiply(mat::Matrix m1, mat::Matrix m2);

//T std(); pas la priorité

}

#endif //INVADERS_MOON_H

#include <vector>
#include <cmath>
#include <chrono>
#include "matrix.h"
#include "data.h"

//Sizes of the intermediate layers are fixed
//Size of the output is with respect to the 
// number of classes in the MNIST dataset
#define LAYER_1_SIZE 120
#define LAYER_2_SIZE 80
#define OUTPUT_SIZE 10

/*
Used to create objects that mimic a flat fully-connected
neural network with 2 middle layers.

Author: ac2255@g.rit.edu
*/
class NeuralNet {
private:
    Matrix input;  
    Matrix layer_1;  
    Matrix layer_2;  

    Matrix weights_input_to_L1;
    Matrix weights_L1_to_L2;
    Matrix weights_L2_to_output;

    Matrix bias_L1;
    Matrix bias_L2;
    Matrix bias_output;

public:
    Matrix output;  

    //Non-parametrized constructor
    NeuralNet(){}

    //Parametrized constructor. The arg "size_t cnn_output_size"
    //comes from the previous cnn layer. It is treated as the size of the 
    //input layer of the current NeuralNet obj being created.
    NeuralNet(size_t cnn_output_size) {
        weights_input_to_L1 = Matrix::initializeRandom({cnn_output_size, LAYER_1_SIZE}, -1.0, 1.0);
        weights_L1_to_L2 = Matrix::initializeRandom({LAYER_1_SIZE, LAYER_2_SIZE}, -1.0, 1.0);
        weights_L2_to_output = Matrix::initializeRandom({LAYER_2_SIZE, OUTPUT_SIZE}, -1.0, 1.0);

        bias_L1 = Matrix::initializeRandom({1, LAYER_1_SIZE}, -1.0, 1.0);
        bias_L2 = Matrix::initializeRandom({1, LAYER_2_SIZE}, -1.0, 1.0);
        bias_output = Matrix::initializeRandom({1, OUTPUT_SIZE}, -1.0, 1.0);
    }


    void forwardPropagation(Matrix inData) {
        
        input = inData;
        layer_1 = (input.matrixMultiply(weights_input_to_L1)).matrixAdd(bias_L1);
        layer_1.relu();  

        layer_2 = (layer_1.matrixMultiply(weights_L1_to_L2)).matrixAdd(bias_L2);
        layer_2.relu(); 

        output = (layer_2.matrixMultiply(weights_L2_to_output)).matrixAdd(bias_output);
        Matrix::sigmoid(&output);  
    }

       
    void backwardPropagation(const Matrix &target, double learningRate) {
    
        Matrix gradient_output = output.matrixSubtract(target);

        Matrix gradient_weights_L2_to_output = layer_2.transpose().matrixMultiply(gradient_output);
        Matrix gradient_bias_output = gradient_output;  

        Matrix gradient_layer_2 = gradient_output.matrixMultiply(weights_L2_to_output.transpose());
        gradient_layer_2 = gradient_layer_2.elementwiseMultiply(layer_2.reluDerivative());

        Matrix gradient_weights_L1_to_L2 = layer_1.transpose().matrixMultiply(gradient_layer_2);
        Matrix gradient_bias_L2 = gradient_layer_2; 

        Matrix gradient_layer_1 = gradient_layer_2.matrixMultiply(weights_L1_to_L2.transpose());
        gradient_layer_1 = gradient_layer_1.elementwiseMultiply(layer_1.reluDerivative());

        Matrix gradient_weights_input_to_L1 = input.transpose().matrixMultiply(gradient_layer_1);
        Matrix gradient_bias_L1 = gradient_layer_1;  

        updateWeights(learningRate, gradient_weights_input_to_L1, gradient_bias_L1, gradient_weights_L1_to_L2, gradient_bias_L2, gradient_weights_L2_to_output, gradient_bias_output);
    }


    void updateWeights(double learningRate, const Matrix &gradient_weights_input_to_L1, const Matrix &gradient_bias_L1, const Matrix &gradient_weights_L1_to_L2, const Matrix &gradient_bias_L2, const Matrix &gradient_weights_L2_to_output, const Matrix &gradient_bias_output) {
        weights_input_to_L1 = weights_input_to_L1.matrixSubtract(gradient_weights_input_to_L1.scalarMultiply(learningRate));
        bias_L1 = bias_L1.matrixSubtract(gradient_bias_L1.scalarMultiply(learningRate));

        weights_L1_to_L2 = weights_L1_to_L2.matrixSubtract(gradient_weights_L1_to_L2.scalarMultiply(learningRate));
        bias_L2 = bias_L2.matrixSubtract(gradient_bias_L2.scalarMultiply(learningRate));

        weights_L2_to_output = weights_L2_to_output.matrixSubtract(gradient_weights_L2_to_output.scalarMultiply(learningRate));
        bias_output = bias_output.matrixSubtract(gradient_bias_output.scalarMultiply(learningRate));
    }

};


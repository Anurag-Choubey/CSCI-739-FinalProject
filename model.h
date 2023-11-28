#include "conv_utils.h"
#include "neuralNet.h"

//Top level declaration of training and testing
// filenames. Make sure that they are in the same dir as your 
// source code files
#define TRAIN_IMAGES_FILE "train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels.idx1-ubyte"
#define TEST_IMAGES_FILE "t10k-images.idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels.idx1-ubyte"

/*
The Model class. 
Orchestrates the entire computation for training and testing.
Has a single convolutional layer, followed by a 3 layered fully-
-connected neural network layer. The class attributes learningRate
and epochs are provided at runtime.

Author: ac2255@g.rit.edu
*/
class Model{
public:
    ConvLayer cnn;
    NeuralNet flat;
    std::vector<MNISTImage> training_data;
    std::vector<MNISTImage> testing_data;
    int epochs; 
    double learningRate;

    //The parametrized constructor
    Model(int filterSize,
        int numFilters, 
        double learning_rate, 
        int epochs ){
        training_data = loadData(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE);
        testing_data = loadData(TEST_IMAGES_FILE, TEST_LABELS_FILE);
        cnn = ConvLayer(training_data[0].rows, static_cast<size_t>(filterSize), static_cast<size_t>(numFilters));
        flat = NeuralNet(cnn.flatSize);
        this->learningRate = learning_rate;
        this->epochs = epochs;
    }

    //Loads the MNIST data from a specified filename into a format that the program requires.
    std::vector<MNISTImage> loadData(const std::string &filenameImgs, const std::string &filenameLbls) {
        std::vector<MNISTImage> images = readImages(filenameImgs, filenameLbls);
        return images;
    }

    //The entire training lifecycle
    void train(){
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();

            std::cout << "EPOCH " << epoch + 1 << std::endl;
            int correctPredictions = 0;

            for (int i = 0; i < training_data.size(); i++) {  
                Matrix input = cnn.forwardPropagation(training_data[i].imageTensor); 
                Matrix target = createTargetMatrix(training_data[i].label);  

                flat.forwardPropagation(input);
                int predictedLabel = flat.output.argmax();
                if (predictedLabel == training_data[i].label) {
                    correctPredictions++;
                }

                flat.backwardPropagation(target, learningRate);
            }

            double accuracy = static_cast<double>(correctPredictions) / training_data.size();
            std::cout << "Accuracy = " << accuracy * 100.0 << "%" << std::endl;

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Epoch " << epoch + 1 << " Time: " << elapsed.count() << " seconds" << std::endl;
            std::cout << "----------------------------------------------------\n";

            if(accuracy * 100.0 > 98.5){
                break;
            }
        }
    }
    
    //Reports test accuracy
    void test() {
        int correctPredictions = 0;
        int totalPredictions = testing_data.size();

        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& image : testing_data) {
            Matrix input = cnn.forwardPropagation(image.imageTensor);
            flat.forwardPropagation(input);
            int predictedLabel = flat.output.argmax();

            if (predictedLabel == image.label) {
                correctPredictions++;
            }
        }
        double accuracy = static_cast<double>(correctPredictions) / totalPredictions;
        std::cout << "Testing Accuracy = " << accuracy * 100.0 << "%" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Testing completed in " << elapsed.count() << " seconds" << std::endl;
    }

    //Creates a one-hot encoding of the actual output label associated with a given input
    Matrix createTargetMatrix(int label) {
        std::vector<double> target(OUTPUT_SIZE, 0.0);
        target[label] = 1.0;  
        return Matrix(target, {1, OUTPUT_SIZE});
    }
};




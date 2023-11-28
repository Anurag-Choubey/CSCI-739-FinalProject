#include <iostream>
#include <algorithm>
#include <omp.h>
#include "matrix.h"

/*
The ConvLayer class. Creates objects that 
have the functionality to perform convolution over an image.
Class attributes filterSize and numFilters are provided at
runtime.

Author: ac2255@g.rit.edu
*/
class ConvLayer {
private:

    size_t input_size;
    std::vector<Matrix> filters; 
    size_t numFilters;
    std::vector<size_t> filterSize; 
    size_t conv_stride;               
    bool useReLU;                 
    size_t poolSize;
    size_t pool_stride;
    

public:
    size_t flatSize;

    //Non-parametrized constructor
    ConvLayer(){};

    //Parametrized constructor
    ConvLayer(size_t input_size, size_t filterSize, size_t numFilters){
        this->input_size = input_size;
        this->filterSize = {filterSize, filterSize};
        this->numFilters = numFilters;
        conv_stride = 1;
        useReLU = true;
        poolSize = 2;
        pool_stride = 2;

        for(size_t i = 0; i < numFilters; ++i){
            filters.push_back(Matrix::initializeRandom(this->filterSize, -1, 1));
        }
        //Used to instantiate the Flat fully-connected NeuralNet object
        calculateFlatSize();
    }

    //Calculate the final dims of the flattened output from the ConvLayer 
    void calculateFlatSize() {
        size_t convOutputRows = (input_size - filterSize[0]) / conv_stride + 1;
        size_t convOutputCols = (input_size - filterSize[1]) / conv_stride + 1;

        size_t poolOutputRows = (convOutputRows - poolSize) / pool_stride + 1;
        size_t poolOutputCols = (convOutputCols - poolSize) / pool_stride + 1;

        flatSize = poolOutputRows * poolOutputCols * numFilters;
    }

    //The actual convolution operation
    Matrix convolve(const Matrix& input, const Matrix& filter) {
        size_t inputRows = input.getDims()[0];
        size_t inputCols = input.getDims()[1];
        size_t filterRows = filter.getDims()[0];
        size_t filterCols = filter.getDims()[1];

        size_t resultRows = ((inputRows - filterRows) / conv_stride) + 1;
        size_t resultCols = ((inputCols - filterCols) / conv_stride) + 1;

        Matrix result({resultRows, resultCols});

        
        // #pragma omp parallel for collapse(2) num_threads(8)
        for (size_t i = 0; i < resultRows; ++i) {
            for (size_t j = 0; j < resultCols; ++j) {
                double sum = 0;
                for (size_t k = 0; k < filterRows; ++k) {
                    for (size_t l = 0; l < filterCols; ++l) {
                        size_t inputRow = i * conv_stride + k;
                        size_t inputCol = j * conv_stride + l;
                        sum += input.getElement(inputRow, inputCol) * filter.getElement(k, l);
                    }
                }
                if (useReLU) {
                    sum = std::max(0.0, sum);
                }
                result.setElement(i, j, sum);
            }
        }
        return result;
    }

    //The pooling operation
    Matrix pool(const std::string& poolType, const Matrix& input) {
        if (poolType != "max" && poolType != "avg") {
            throw std::runtime_error("Unknown pooling type. Use \"max\" or \"avg\" ");
        }

        size_t mxPoolDimX = ((input.getDims()[0] - poolSize) / pool_stride) + 1;
        size_t mxPoolDimY = ((input.getDims()[1] - poolSize) / pool_stride) + 1;

        Matrix result({mxPoolDimX, mxPoolDimY});

        for (size_t i = 0; i < mxPoolDimX; ++i) {
            for (size_t j = 0; j < mxPoolDimY; ++j) {
                std::vector<double> pooler;
                for (size_t k = i * pool_stride; k < (i * pool_stride) + poolSize; ++k) {
                    for (size_t l = j * pool_stride; l < (j * pool_stride) + poolSize; ++l) {
                        pooler.push_back(input.getElement(k, l));
                    }
                }
                double poolValue = 0.0;
                if (poolType == "max") {
                    poolValue = *std::max_element(pooler.begin(), pooler.end());
                } else if (poolType == "avg") {
                    double sum = std::accumulate(pooler.begin(), pooler.end(), 0.0);
                    poolValue = sum / static_cast<double>(pooler.size());
                }
                result.setElement(i, j, poolValue);
            }
        }
        return result;
    }

    Matrix forwardPropagation(const Matrix& input){
        std::vector<Matrix> conv_pool_ops(numFilters);

        #pragma omp parallel for num_threads(8)
        for(int  i = 0; i < numFilters; i++){
            Matrix tmp = convolve(input, filters[i]);
            Matrix tmp2 = pool("max", tmp);
            #pragma omp critical
                conv_pool_ops[i] = tmp2;
        }

        Matrix result = Matrix::flattenMatrices(conv_pool_ops);
        return result;
    }

};
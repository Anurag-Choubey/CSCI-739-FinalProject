#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <stdexcept> 

/*
The Matrix class.
The backbone of the entire learning system.
All collective data movement is done as an instance 
of this class. Every operation of the flat layers is
expressed as the method calls to this class.

Author: ac2255@g.rit.edu
*/
class Matrix {
private:
    //The actual data of the matrix
    std::vector<double> data;
    //The dimensions of the matrix
    std::vector<size_t> dims;

    size_t calculateTotalSize(const std::vector<size_t>& dimensions) const {
        size_t totalSize = 1;
        for (size_t dim : dimensions) {
            totalSize *= dim;
        }
        return totalSize;
    }

public:
    Matrix(){}

    Matrix(std::vector<size_t> dims) : dims(dims){
        data.resize(calculateTotalSize(dims));
    }


    Matrix(std::vector<double> data, std::vector<size_t> dims) : data(data), dims(dims) {
        if (data.size() != calculateTotalSize(dims)) {
            throw std::invalid_argument("Data size doesn't match specified dimensions.");
        }
    }


    std::vector<double>& getData() {
        return data;
    }


    const std::vector<size_t> getDims() const {
        return dims;
    }


    void setElement(size_t row, size_t col, double value) {
        if (row >= dims[0] || col >= dims[1]) {
            throw std::out_of_range("Matrix indices out of range.");
        }
        data[row * dims[1] + col] = value;
    }


    inline double getElement(size_t row, size_t col) const {
        if (row >= dims[0] || col >= dims[1]) {
            throw std::out_of_range("Matrix indices out of range.");
        }
        return data[row * dims[1] + col];
    }


    void normalizeWith(double val){
         for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                double elem = this->getElement(i, j);
                this->setElement(i,j, elem / val);
            }
         }
    }


    int argmax() {
        if (this->getDims()[0] != 1) {
            throw std::invalid_argument("Invalid matrix dimensions for argmax. Expected a row vector.");
        }
        int maxIdx = 0;
        double maxVal = data[0]; 

        for (size_t i = 1; i < this->getDims()[1]; ++i) {
            if (data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }


    static Matrix zeros(const std::vector<size_t>& dimensions) {
        size_t totalSize = 1;
        for (size_t dim : dimensions) {
            totalSize *= dim;
        }

        std::vector<double> zeroData(totalSize, 0.0);

        return Matrix(zeroData, dimensions);
    }


    static void sigmoid(Matrix* matPtr) {
        if (matPtr->getDims()[0] != 1) {
            throw std::invalid_argument("Invalid matrix dimensions for sigmoid. Expected a row vector.");
        }
        for (size_t i = 0; i < matPtr->getDims()[1]; ++i) {
            double val = matPtr->getElement(0, i);
            val = 1.0 / (1.0 + exp(-val));
            matPtr->setElement(0, i, val);
        }
    }


    Matrix sigmoidDerivative() const {
        std::vector<double> derivativeData(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            double sigmoidValue = 1.0 / (1.0 + exp(-data[i]));  
            derivativeData[i] = sigmoidValue * (1.0 - sigmoidValue);  
        }

        return Matrix(derivativeData, dims);
    }

    
    void relu() {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::max(0.0, data[i]);
        }
    }


    Matrix reluDerivative() const {
        Matrix derivative(this->getDims());  

        for (size_t i = 0; i < this->data.size(); ++i) {
            derivative.data[i] = this->data[i] > 0 ? 1.0 : 0.0;
        }

        return derivative;
    }


    Matrix transpose() const {
        std::vector<size_t> transposedDims = {dims[1], dims[0]};  
        std::vector<double> transposedData(transposedDims[0] * transposedDims[1]);

        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                transposedData[j * dims[0] + i] = data[i * dims[1] + j];
            }
        }

        return Matrix(transposedData, transposedDims);
    }

    Matrix flatten() const {
        std::vector<double> flatData = data; 
        std::vector<size_t> flatDims = {1, data.size()}; 
        return Matrix(flatData, flatDims);
    }

    static Matrix flattenMatrices(const std::vector<Matrix>& matrices) {
        std::vector<double> combinedData;

        for (const auto& mat : matrices) {
            Matrix flatMat = mat.flatten(); 
            combinedData.insert(combinedData.end(), flatMat.data.begin(), flatMat.data.end());
        }

        return Matrix(combinedData, {1, combinedData.size()});
    }


    Matrix matrixMultiply(const Matrix& other) const {
        if (dims[1] != other.dims[0]) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
        }
        
        std::vector<double> resultData(dims[0] * other.dims[1], 0.0);
        std::vector<size_t> resultDims = { dims[0], other.dims[1] };

        // #pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t k = 0; k < dims[1]; ++k) {
                if(data[i * dims[1] + k] != 0){
                    for (size_t j = 0; j < other.dims[1]; ++j) {
                        resultData[i * resultDims[1] + j] += data[i * dims[1] + k] * other.data[k * other.dims[1] + j];
                    }
                }
            }
        }
        return Matrix(resultData, resultDims);
    }


    Matrix matrixAdd(const Matrix& other) const {
        if (dims != other.dims) {
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        }
        std::vector<double> resultData(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            resultData[i] = data[i] + other.data[i];
        }
        return Matrix(resultData, dims);
    }


    Matrix matrixSubtract(const Matrix& other) const {
        if (dims != other.dims) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction.");
        }
        std::vector<double> resultData(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            resultData[i] = data[i] - other.data[i];
        }
        return Matrix(resultData, dims);
    }


    Matrix elementwiseMultiply(const Matrix& other) const {
        if (dims != other.dims) {
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication.");
        }

        std::vector<double> resultData(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            resultData[i] = data[i] * other.data[i];
        }

        return Matrix(resultData, dims);
    }


    Matrix scalarMultiply(double scalar) const {
        std::vector<double> scaledData(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            scaledData[i] = data[i] * scalar;
        }

        return Matrix(scaledData, dims);
    }


    static Matrix initializeRandom(const std::vector<size_t>& dimensions, double minVal, double maxVal) {
        std::vector<double> randomData;
        randomData.reserve(dimensions[0] * dimensions[1]);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(minVal, maxVal);

        for (size_t i = 0; i < (dimensions[0] * dimensions[1]); ++i) {
            randomData.push_back(dis(gen));
        }
        return Matrix(randomData, dimensions);
    }


    void printMatrix() const {
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                std::cout << data[i * dims[1] + j] << "\t";
            }
            std::cout << std::endl;
        }
    }
};

#endif

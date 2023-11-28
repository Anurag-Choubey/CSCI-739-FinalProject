#ifndef DATA_H  
#define DATA_H  

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept> 
#include "matrix.h"

/*
Utility to read the MNIST files
as provided on http://yann.lecun.com/exdb/mnist/

Author: ac2255@g.rit.edu
*/

//Storage for each image, along with relevant fields
struct MNISTImage {
    size_t rows;
    size_t cols;
    Matrix imageTensor;
    int label;
};

//Load the file and return a vector of images represented as a struct
std::vector<MNISTImage> readImages(const std::string &filenameImgs, const std::string &filenameLbls) {
    std::vector<MNISTImage> images;

    std::ifstream fileImages(filenameImgs, std::ios::binary);
    if (!fileImages.is_open()) {
        std::cerr << "Failed to open file: " << filenameImgs << std::endl;
        return images;
    }

    uint32_t magic1, numImages, numRows, numCols;
    fileImages.read(reinterpret_cast<char*>(&magic1), sizeof(magic1));
    fileImages.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    fileImages.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    fileImages.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    
    magic1 = __builtin_bswap32(magic1);
    numImages = __builtin_bswap32(numImages);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    if (numCols > 28 || numRows > 28){
        std::cerr << "Input image dimensions not as per MNIST spec." << std::endl;
    }

    std::vector<int> labels;

    std::ifstream fileLabel(filenameLbls, std::ios::binary);
    if (!fileLabel.is_open()) {
        std::cerr << "Failed to open file: " << filenameLbls << std::endl;
        return images;
    }

    //Top values read here
    uint32_t magic2, numLabels;
    fileLabel.read(reinterpret_cast<char*>(&magic2), sizeof(magic2));
    fileLabel.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    magic2 = __builtin_bswap32(magic2);
    numLabels = __builtin_bswap32(numLabels);

    std::vector<u_int8_t> rawData;
    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t label;
        fileLabel.read(reinterpret_cast<char*>(&label), sizeof(label));
        rawData.push_back(label);
    }

    std::vector<int> intVector(rawData.begin(), rawData.end());
    labels.assign(intVector.begin(), intVector.end());

    if(numImages != numLabels){
        throw std::invalid_argument("Images size and label size mismatch.");
    }

    for (uint32_t i = 0; i < numImages; ++i) {
        MNISTImage mnist_img;
        mnist_img.rows = static_cast<size_t>(numRows);
        mnist_img.cols = static_cast<size_t>(numCols);

        std::vector<u_int8_t> rawData; rawData.resize(numRows * numCols);
        fileImages.read(reinterpret_cast<char*>(rawData.data()), rawData.size());

        std::vector<double> dubVector(rawData.begin(), rawData.end());

        mnist_img.imageTensor = Matrix(dubVector, {mnist_img.rows , mnist_img.cols});
        mnist_img.imageTensor.normalizeWith(255.0);
        mnist_img.label = labels[i];
        if (mnist_img.label < 0 || mnist_img.label > 9){
            std::cerr << "Illegal label value as per MNIST spec." << std::endl;
        }

        images.push_back(mnist_img);
    }

    fileImages.close();  
    fileLabel.close();   

    return images;
}

#endif
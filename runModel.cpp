#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "model.h"

/*
The driver program.
Takes in  the comand-line args and creates the model object.
Subsequently it calls the train and test methods respectively.

Author: ac2255@g.rit.edu
*/
int main( int argc, char* argv[] ) {

  if (argc != 5) {
        std::cerr << "Usage: ./runModel filterSize numFilters learning_rate epochs\n";
        return 1;
    }

    try {
        int filterSize = std::stoi(argv[1]); 
        int numFilters = std::stoi(argv[2]); 
        double learning_rate = std::stod(argv[3]); 
        int epochs = std::stoi(argv[4]); 

        Model miniCon = Model(filterSize, numFilters, learning_rate, epochs);
        miniCon.train();
        miniCon.test();
        return 0;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
        return 1;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Argument out of range: " << oor.what() << '\n';
        return 1;
    }

 
  
}

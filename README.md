Final Project
CSCI 739: Machine Learning Systems Implementation, Fall 2023

The associated source code contains all the utilities to train and test a light
convolutional neural network on the MNIST dataset as provided at http://yann.lecun.com/exdb/mnist/


1. The Architecture
    > The model consists of a single convolutional layer. The kernel size
    and number of kernels that will be involved in the convolution can be set by the user at runtime

    > The outputs from the convolution are then pooled to reduce the size of the feature maps.
    Although not provided at runtime, the code may be changed in the forwardProp method in 
    the conv_utils.h file to switch between average and max pooling. The default setting is 
    max pooling.

    > The feature maps are then subjet to ReLu activation before being passed to pooling layer.

    > The multiple feature maps generated are then flattened and fed to a dense fully connected
    neural network, with two middle layers. The sizes of the middle layes are kept constant at 
    120 and 80 , while the size of the output layer is 10, in order to match the total number
    of classes that an MNIST sample may fall into.

    > The middle layers are of the particular size because optimal performance could be
    reached with these hyperparameters.

    > The middle layers use ReLu activation for output, whereas the output layer uses softmax
    in order to represent the final result as a matrix of probabilities.

2. How to run the code

    > The code currently is compiled using clang++. So users of Mac and Linux must
    make sure that they have a stable version of clang++ installed on their system.
    The code compiles and runs successfully on the author's MacBook Pro, as well as 
    the granger system @ Rochester Institute of Technology, which is a Linux system.

    > To execute, make sure that the code as well as the data files all reside in the same dir,
    and at the same level within the dir.

    > Execute with:
        Mac - zsh compile.sh && zsh run.sh
        Linux - bash compile.sh && bash run.sh

    > Command Line Args - They can be provided in the run.sh file.
    Usage: ./runModel
                 filterSize
                     numFilters
                         learning_rate
                                     epochs
    Currently the params are set to ./runModel 6 8 0.00125 75, as they allow fo the 
    most optimal training.


3. Evaluation and Benchmarking
    > Each epoch of the model takes approximately 30 seconds to train on the 
    author's local machine, ie , a MacBook Pro with Apple silicon. The training
    time per epoch on the granger system is 57 seconds.

    > While the number of epochs can be set by the user. The model is designed to
    automatically cut-off the training if the training accuracy crosses 98.5%. Any 
    further training beyond this point results in a drop in testing accuracy and therefore an
    increase in test error rate.

    > Our model converges to training 98.5% accuracy in 35 +- 3 epochs. The testing accuracy 
    achieved is 96.5 +- 0.2 %. The average test error reate then may be reported as being 3.5% on the
    MNIST 10k testing dataset.

    > Here are some models with their test error rates for comparison/benchmarking:
         Model      : Test Error Rate (%)
        LeNet-1     :     1.7 
        LeNet-5     :     0.95 
        LeNet-4     :     1.1


4. Remarks
    > The code can run only on a CPU and is not parallelized for GPU's.
    > The convolution of each image with multiple filters has been 
    parallelized using OpenMP. 
    > When the Matrix operations such as matrixMultiply and matrixAdd
    were parallelized using OpenMP, the training time per epoch actualy went up
    from 30 seconds to 46 seconds on the author's local machine. This can only 
    be attributed to the large overhead involved in each of the 60,000 iterations
    per epoch.
    > Similarly, parallelization of the convolution and pooling operations had also 
    resulted in a drop in performance (measured as time/epoch). 
    

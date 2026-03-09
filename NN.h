#ifndef NN_H
#define NN_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

struct neuron {
    /// @brief stores the neurons bias
    float bias;

    /// @brief stores the weights for the neuron
    std::vector<float> weights;

    /// @brief stores the derivative of each weight with respect to the cost function, calculated during back propagation
    std::vector<float> weightGradients;

    /// @brief stores the derivative of each weight with respect to the cost function, before multiplying by the output of the prior layer
    float delta;

    /// @brief stores the biases gradient
    float biasGradient;

    /// @brief used by RMSProp during optimization
    std::vector<float> velocity;

    /// @brief used by ADAM during optimization
    std::vector<float> momentum;

    /// @brief used by RPSProp during optimization
    float biasVelocity;

    /// @brief used by ADAM during optimization
    float biasMomentum;

    /// @brief allocate space in the vector for all the weights
    /// @param count amount of weights the neuron has
    void allocateWeights(int count);

    /// @brief calculate the neurons output
    /// @param inputs a pointer to the array that contains the inputs for the neuron
    /// @return the neurons output

    /// @brief calculate the neurons output
    /// @param inputs a pointer to the array that contains the inputs for the neuron
    /// @param activationFunction a pointer to the activation function
    /// @return the neurons output
    float calculate(float* inputs, float (*activationFunction)(float));

    /// @brief calculate the neurons output without an activation function, useful for output layers
    /// @param inputs a pointer to the array that contains the inputs for the neuron
    /// @return the neurons output
    float calculateWithoutActivation(float* inputs);

    /// @brief initialize all of the weights in the neuron to random values
    /// @param randomEngine a rng that will be used with the normal distrobution to set the parameters
    /// @param normalDistro a normal distrobution that will be used with the rng to set the parameters
    void initialize(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief add a random number to all of the weights in the neuron
    /// @param randomEngine a rng that will be used with the normal distrobution to add to the parameters
    /// @param normalDistro a normal distrobution that will be used with the rng to add to the parameters
    void mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief prints out the weights and bias of the neuron
    void print();

    /// @brief prints out the weights and bias of the neuron to a file
    void printToFile(std::ofstream& file);
};

struct layer {
    /// @brief used to store the outputs of each neuron for easy access
    std::vector<float> output;

    std::vector<float> noActOutput;

    /// @brief stores the neurons to form the layer
    std::vector<neuron> neurons;

    /// @brief allocate space in the vector for all the neurons
    /// @param count amount of neurons the layer has
    void allocateNeurons(int count);

    /// @brief calculates each neurons output and stores it in the output vector
    /// @param inputs a pointer to the array that contains the inputs for the layer
    /// @param activationFunction a pointer to the activation function
    void calculate(float* inputs, float (*activationFunction)(float));

    /// @brief calculates each neurons output without the activation function and stores it in the output vector
    /// @param inputs a pointer to the array that contains the inputs for the layer
    void calculateWithoutActivation(float* inputs);

    /// @brief calculates each neurons output and stores it in the output vector, also stores non activated outputs in an array
    /// @param inputs a pointer to the array that contains the inputs for the layer
    /// @param activationFunction a pointer to the activation function
    void trainingCalculate(float* inputs, float (*activationFunction)(float));

    /// @brief calculates each neurons output without the activation function and stores it in the output vector, also stores non activated outputs in an array
    /// @param inputs a pointer to the array that contains the inputs for the layer
    void trainingCalculateWithoutActivation(float* inputs);

    /// @brief initialize all of the weights in all of the neurons in the layer to random values
    /// @param randomEngine a rng that will be used with the normal distrobution to set the parameters of the neurons in the layer
    /// @param normalDistro a normal distrobution that will be used with the rng to set the parameters of the neurons in the layer
    void initialize(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief add a random number to all of the eights in all of the neurons in the layer
    /// @param randomEngine a rng that will be used with the normal distrobution to add to the parameters of the neurons in the layer
    /// @param normalDistro a normal distrobution that will be used with the rng to add to the parameters of the neurons in the layer
    void mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief prints out the weights and biases of the neurons in the layer
    void print();

    /// @brief prints out the weights and biases of the neurons in the layer to a file
    void printToFile(std::ofstream& file);
};

struct network {
    /// @brief stores the layers to form the network
    std::vector<layer> layers;

    /// @brief allocate space in the vector for all the layers
    /// @param count amount of layers in the network
    void allocateLayers(int count);

    /// @brief constructs a network with the specified layer sizes
    /// @param layerSizes the size of each layer, index 0 is the number of input neurons and won't be a stored layer, the last index is the number of output neurons
    /// @param layerCount number of layers in the network
    void allocateNetwork(int* layerSizes, int layerCount);

    /// @brief calculates each layer and feeds their outputs to the inputs of the next layer
    /// @param input a pointer to the array that contains the inputs for the network
    /// @param activationFunction a pointer to the activation function to use
    /// @param outputActivationFunction a pointer to the activation function to use for the outputlayer
    void calculate(float* input, float (*activationFunction)(float), float (*outputActivationFunction)(float));

    /// @brief calculates each layer and feeds their outputs to the inputs of the next layer without activations on the last layer
    /// @param input a pointer to the array that contains the inputs for the network
    /// @param activationFunction a pointer to the activation function to use
    void calculate(float* input, float (*activationFunction)(float));

    /// @brief calculates each layer and feeds their outputs to the inputs of the next layer
    /// @param input a pointer to the array that contains the inputs for the network
    /// @param activationFunction a pointer to the activation function to use
    /// @param outputActivationFunction a pointer to the activation function to use for the outputlayer
    void trainingCalculate(float* input, float (*activationFunction)(float), float (*outputActivationFunction)(float));

    /// @brief calculates each layer and feeds their outputs to the inputs of the next layer without activations on the last layer
    /// @param input a pointer to the array that contains the inputs for the network
    /// @param activationFunction a pointer to the activation function to use
    void trainingCalculate(float* input, float (*activationFunction)(float));

    /// @brief uses back propagation to calculate the derivative of the cost function in relation to each weight and bias
    /// @param input the input to calculate the forward pass with
    /// @param target pointer to the output layers target values
    /// @param activationFunction pointer to the hidden layers activation function
    /// @param activationFunctionDerivative pointer to the hidden layers activation functions derivative
    /// @param outputActivationFunction pointer to the output layer activation function
    /// @param outputActivationFunctionDerivative pointer to the output layer activation functions derivative
    /// @param costFunction pointer to the cost function
    /// @param costFunctionDerivative pointer to the cost functions derivative
    void backPropagation(float* input, float* target, float (*activationFunction)(float), float (*activationFunctionDerivative)(float), float (*outputActivationFunction)(float), float (*outputActivationFunctionDerivative)(float), float (*costFunction)(float, float), float (*costFunctionDerivative)(float, float));

    /// @brief uses back propagation to calculate the derivative of the cost function in relation to each weight and bias
    /// @param input the input to calculate the forward pass with
    /// @param target pointer to the output layers target values
    /// @param activationFunction pointer to the hidden layers activation function
    /// @param activationFunctionDerivative pointer to the hidden layers activation functions derivative
    /// @param costFunction pointer to the cost function
    /// @param costFunctionDerivative pointer to the cost functions derivative
    void backPropagation(float* input, float* target, float (*activationFunction)(float), float (*activationFunctionDerivative)(float), float (*costFunction)(float, float), float (*costFunctionDerivative)(float, float));

    /// @brief uses stochastic gradient descent to optimize the model
    /// @param learningRate learning rate for the neurons
    void SGD(float learningRate);

    /// @brief uses root mean squared propagation to optimise the model
    /// @param learningRate learning rate for the neurons
    /// @param beta between 0 and 1, a value of 0 will only use current readings, 1 will only use past readings
    void RMSProp(float learningRate, float beta = 0.9);

    /// @brief uses adaptive moment estimation to optimize the model
    /// @param learningRate learning rate for the neurons
    /// @param beta between 0 and 1, a value of 0 will only use current readings, 1 will only use past readings
    /// @param beta2 same as beta, but used for past velocity rather than inputs
    void ADAM(float learningRate, float beta = 0.9, float beta2 = 0.999);

    /// @brief initialize all of the weights in all of the neurons in the network to random values
    /// @param randomEngine a rng that will be used with the normal distrobution to set the parameters of the neurons in the layer
    /// @param normalDistro a normal distrobution that will be used with the rng to set the parameters of the neurons in the layer
    void initialize(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief add a random number to all of the weights in all of the neurons in the network
    /// @param randomEngine a rng that will be used with the normal distrobution to add to the parameters of the neurons in the layer
    /// @param normalDistro a normal distrobution that will be used with the rng to add to the parameters of the neurons in the layer
    void mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro);

    /// @brief prints out the weights and biases of the neurons in the network
    void print();

    /// @brief prints out the weights and biases of the neurons in the network to a file
    void printToFile(std::ofstream& file);
};

/// @brief constructs a network struct from a file
/// @param fileName the name of the file to read from
/// @param output the pointer to the network to write to
/// @return the constructed network
void readNetworkFromFile(std::string fileName, network* output);

/// @brief writes a network to a file
/// @param fileName the name of the file to write to
/// @param input the network to read from
void writeNetworkToFile(std::string fileName, network input);

/// @brief ReLU activation function
/// @param input the variable to apply the function to
/// @return the functions output
float ReLU(float input);
#endif
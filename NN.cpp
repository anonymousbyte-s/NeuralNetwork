#include "NN.h"

// NEURON

void neuron::allocateWeights(int count) {
    weights.resize(count);
    weightGradients.resize(count);
    velocity.resize(count);
    momentum.resize(count);
}

float neuron::calculate(float* inputs, float (*activationFuncton)(float)) {
    float output = 0;
    for (int i = 0; i < weights.size(); i++) {
        output += inputs[i] * weights[i];
    }
    return (activationFuncton(output + bias));
}

float neuron::calculateWithoutActivation(float* inputs) {
    float output = 0;
    for (int i = 0; i < weights.size(); i++) {
        output += inputs[i] * weights[i];
    }
    return (output + bias);
}

void neuron::initialize(std::default_random_engine& randomEngine, float variance) {
    std::normal_distribution<float> normalDistro(0.0, variance);
    for (int i = 0; i < weights.size(); i++) {
        weights[i] = normalDistro(randomEngine);
    }
    bias = 0;
}

void neuron::mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro) {
    for (int i = 0; i < weights.size(); i++) {
        weights[i] += normalDistro(randomEngine);
        weights[i] = weights[i];
    }
    bias += normalDistro(randomEngine);
}

void neuron::print() {
    std::cout << "Weights: \n";
    for (int i = 0; i < weights.size(); i++) {
        std::cout << weights[i] << '\n';
    }
    std::cout << "Bias: \n"
              << bias << '\n';
}

void neuron::printToFile(std::ofstream& file) {
    file << "Weights: \n";
    for (int i = 0; i < weights.size(); i++) {
        file << weights[i] << '\n';
    }
    file << "Bias: \n"
         << bias << '\n';
}

// NETWORK LAYER

void layer::allocateNeurons(int count) {
    neurons.resize(count);
    output.resize(count);
    noActOutput.resize(count);
}

void layer::calculate(float* inputs, float (*activationFuncton)(float)) {
    for (int i = 0; i < neurons.size(); i++) {
        output[i] = neurons[i].calculate(inputs, activationFuncton);
    }
}

void layer::calculateWithoutActivation(float* inputs) {
    for (int i = 0; i < neurons.size(); i++) {
        output[i] = neurons[i].calculateWithoutActivation(inputs);
    }
}

void layer::trainingCalculate(float* inputs, float (*activationFuncton)(float)) {
    for (int i = 0; i < neurons.size(); i++) {
        noActOutput[i] = neurons[i].calculateWithoutActivation(inputs);
        output[i] = activationFuncton(noActOutput[i]);
    }
}

void layer::trainingCalculateWithoutActivation(float* inputs) {
    for (int i = 0; i < neurons.size(); i++) {
        noActOutput[i] = neurons[i].calculateWithoutActivation(inputs);
        output[i] = noActOutput[i];
    }
}

void layer::initialize(std::default_random_engine& randomEngine, float variance) {
    for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++) {
        neurons[neuronIndex].initialize(randomEngine, variance);
    }
}

void layer::mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro) {
    for (int neuronIndex = 0; neuronIndex < neurons.size(); neuronIndex++) {
        neurons[neuronIndex].mutate(randomEngine, normalDistro);
    }
}

void layer::print() {
    for (int i = 0; i < neurons.size(); i++) {
        std::cout << "Neuron " << i << '\n';
        neurons[i].print();
        std::cout << '\n';
    }
}

void layer::printToFile(std::ofstream& file) {
    for (int i = 0; i < neurons.size(); i++) {
        file << "Neuron " << i << '\n';
        neurons[i].printToFile(file);
        file << '\n';
    }
}

// NETWORK

void network::allocateLayers(int count) {
    layers.resize(count);
}

void network::allocateNetwork(int* layerSizes, int layerCount) {
    allocateLayers(layerCount);
    // layers
    for (int layerIndex = 0; layerIndex < layerCount; layerIndex++) {
        layers[layerIndex].allocateNeurons(layerSizes[layerIndex + 1]);
        // neurons
        for (int neuronIndex = 0; neuronIndex < layerSizes[layerIndex + 1]; neuronIndex++) {
            layers[layerIndex].neurons[neuronIndex].allocateWeights(layerSizes[layerIndex]);
        }
    }
}

void network::calculate(float* input, float (*activationFunction)(float), float (*outputActivationFunction)(float)) {
    // if the network only has one layer, calculate the output with the output layer activation function and exit
    if (layers.size() == 1) {
        layers[0].calculate(input, outputActivationFunction);
        return;
    }
    // if there is more than one layer, calculate the output of the first layer with the activation
    layers[0].calculate(input, activationFunction);
    for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
        // if the last layer is being calculated, use the output layer activation
        if (layerIndex == layers.size() - 1) {
            layers[layerIndex].calculate(layers[layerIndex - 1].output.data(), outputActivationFunction);
            continue;
        }
        // give the last layers output to the current layers inputs
        layers[layerIndex].calculate(layers[layerIndex - 1].output.data(), activationFunction);
    }
}

void network::calculate(float* input, float (*activationFunction)(float)) {
    // if the network only has one layer, calculate the output with the output layer activation function and exit
    if (layers.size() == 1) {
        layers[0].calculateWithoutActivation(input);
        return;
    }
    // if there is more than one layer, calculate the output of the first layer with the activation
    layers[0].calculate(input, activationFunction);
    for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
        // if the last layer is being calculated, use the output layer activation
        if (layerIndex == layers.size() - 1) {
            layers[layerIndex].calculateWithoutActivation(layers[layerIndex - 1].output.data());
            continue;
        }
        // give the last layers output to the current layers inputs
        layers[layerIndex].calculate(layers[layerIndex - 1].output.data(), activationFunction);
    }
}

void network::trainingCalculate(float* input, float (*activationFunction)(float), float (*outputActivationFunction)(float)) {
    // if the network only has one layer, calculate the output with the output layer activation function and exit
    if (layers.size() == 1) {
        layers[0].trainingCalculate(input, outputActivationFunction);
        return;
    }
    // if there is more than one layer, calculate the output of the first layer with the activation
    layers[0].trainingCalculate(input, activationFunction);
    for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
        // if the last layer is being calculated, use the output layer activation
        if (layerIndex == layers.size() - 1) {
            layers[layerIndex].trainingCalculate(layers[layerIndex - 1].output.data(), outputActivationFunction);
            continue;
        }
        // give the last layers output to the current layers inputs
        layers[layerIndex].trainingCalculate(layers[layerIndex - 1].output.data(), activationFunction);
    }
}

void network::trainingCalculate(float* input, float (*activationFunction)(float)) {
    // if the network only has one layer, calculate the output with the output layer activation function and exit
    if (layers.size() == 1) {
        layers[0].trainingCalculateWithoutActivation(input);
        return;
    }
    // if there is more than one layer, calculate the output of the first layer with the activation
    layers[0].trainingCalculate(input, activationFunction);
    for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
        // if the last layer is being calculated, use the output layer activation
        if (layerIndex == layers.size() - 1) {
            layers[layerIndex].trainingCalculateWithoutActivation(layers[layerIndex - 1].output.data());
            continue;
        }
        // give the last layers output to the current layers inputs
        layers[layerIndex].trainingCalculate(layers[layerIndex - 1].output.data(), activationFunction);
    }
}

void network::backPropagation(float* input, float* target, float (*activationFunction)(float), float (*activationFunctionDerivative)(float), float (*outputActivationFunction)(float), float (*outputActivationFunctionDerivative)(float), float (*costFunction)(float, float), float (*costFunctionDerivative)(float, float)) {
    // run the forward pass to store the outputs of each layer without the activation functions applied
    trainingCalculate(input, activationFunction, outputActivationFunction);

    // calculate the deltas for the entire network
    // iterate through all the layers
    for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--) {
        // iterate through all the neurons in the current layer
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            // reset the delta before using it
            layers[layerIndex].neurons[neuronIndex].delta = 0;
            // on the last layer, use the cost function derivative to find the delta
            if (layerIndex == layers.size() - 1) {
                layers[layerIndex].neurons[neuronIndex].delta = costFunctionDerivative(layers[layerIndex].output[neuronIndex], target[neuronIndex]) * outputActivationFunctionDerivative(layers[layerIndex].noActOutput[neuronIndex]);
                continue;
            }
            // iterate through all the outputs of the next layer (the layer closer to the output layer)
            for (int nextNeuronIndex = 0; nextNeuronIndex < layers[layerIndex + 1].neurons.size(); nextNeuronIndex++) {
                // sum all the derivatives of the weights from the next layer
                layers[layerIndex].neurons[neuronIndex].delta += layers[layerIndex + 1].neurons[nextNeuronIndex].delta * layers[layerIndex + 1].neurons[nextNeuronIndex].weights[neuronIndex];
            }
            // multiply this sum by the derivative of the cost with respect to the current layers output, without activation since the derivative of the activation function is being used
            layers[layerIndex].neurons[neuronIndex].delta *= activationFunctionDerivative(layers[layerIndex].noActOutput[neuronIndex]);
        }
    }

    // calculate the final gradients of all the weights
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                // when calculating the first layers gradients, use the networks inputs since there is no other layer before the first one
                if (layerIndex == 0) {
                    layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] += input[weightIndex] * layers[layerIndex].neurons[neuronIndex].delta;
                    continue;
                }
                layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] += layers[layerIndex - 1].output[weightIndex] * layers[layerIndex].neurons[neuronIndex].delta;
            }
            layers[layerIndex].neurons[neuronIndex].biasGradient += layers[layerIndex].neurons[neuronIndex].delta;
        }
    }
}

void network::backPropagation(float* input, float* target, float (*activationFunction)(float), float (*activationFunctionDerivative)(float), float (*costFunction)(float, float), float (*costFunctionDerivative)(float, float)) {
    // run the forward pass to store the outputs of each layer without the activation functions applied
    trainingCalculate(input, activationFunction);

    // calculate the deltas for the entire network
    // iterate through all the layers
    for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--) {
        // iterate through all the neurons in the current layer
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            // reset the delta before using it
            layers[layerIndex].neurons[neuronIndex].delta = 0;
            // on the last layer, use the cost function derivative to find the delta
            if (layerIndex == layers.size() - 1) {
                layers[layerIndex].neurons[neuronIndex].delta = costFunctionDerivative(layers[layerIndex].output[neuronIndex], target[neuronIndex]);
                continue;
            }
            // iterate through all the outputs of the next layer (the layer closer to the output layer)
            for (int nextNeuronIndex = 0; nextNeuronIndex < layers[layerIndex + 1].neurons.size(); nextNeuronIndex++) {
                // sum all the derivatives of the weights from the next layer
                layers[layerIndex].neurons[neuronIndex].delta += layers[layerIndex + 1].neurons[nextNeuronIndex].delta * layers[layerIndex + 1].neurons[nextNeuronIndex].weights[neuronIndex];
            }
            // multiply this sum by the derivative of the cost with respect to the current layers output, without activation since the derivative of the activation function is being used
            layers[layerIndex].neurons[neuronIndex].delta *= activationFunctionDerivative(layers[layerIndex].noActOutput[neuronIndex]);
        }
    }

    // calculate the final gradients of all the weights
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                // when calculating the first layers gradients, use the networks inputs since there is no other layer before the first one
                if (layerIndex == 0) {
                    layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] += input[weightIndex] * layers[layerIndex].neurons[neuronIndex].delta;
                    continue;
                }
                layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] += layers[layerIndex - 1].output[weightIndex] * layers[layerIndex].neurons[neuronIndex].delta;
            }
            layers[layerIndex].neurons[neuronIndex].biasGradient += layers[layerIndex].neurons[neuronIndex].delta;
        }
    }
}

void network::addGradients(const std::vector<network>& inputNetwork) {
    float sizeInverse = 1.0 / inputNetwork.size();
    // iterate through all the layers and sum the gradients
    for (size_t networkIndex = 0; networkIndex < inputNetwork.size(); networkIndex++) {
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            // iterate through all the neurons in each layer
            for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
                // iterate through all the weights in each layer
                for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                    // add the input networks neurons gradients to the current networks gradients
                    layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] += inputNetwork[networkIndex].layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex];
                }
                // add the input networks neurons bias gradient to the current networks bias gradient
                layers[layerIndex].neurons[neuronIndex].biasGradient += inputNetwork[networkIndex].layers[layerIndex].neurons[neuronIndex].biasGradient;
            }
        }
    }
    // iterate through all the layers and average the gradients
    for (size_t networkIndex = 0; networkIndex < inputNetwork.size(); networkIndex++) {
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            // iterate through all the neurons in each layer
            for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
                // iterate through all the weights in each layer
                for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                    // add the input networks neurons gradients to the current networks gradients
                    layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] *= sizeInverse;
                }
                // add the input networks neurons bias gradient to the current networks bias gradient
                layers[layerIndex].neurons[neuronIndex].biasGradient *= sizeInverse;
            }
        }
    }
}

void network::SGD(float learningRate) {
    // iterate through all the layers
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        // iterate through all the neurons in each layer
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            // iterate through all the neurons
            for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                // iterate through all the weights and change them
                layers[layerIndex].neurons[neuronIndex].weights[weightIndex] -= learningRate * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex];
                layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] = 0;
            }
            // change the bias by the bias gradients
            layers[layerIndex].neurons[neuronIndex].bias -= learningRate * layers[layerIndex].neurons[neuronIndex].biasGradient;
            layers[layerIndex].neurons[neuronIndex].biasGradient = 0;
        }
    }
}

void network::RMSProp(float learningRate, float beta) {
    // iterate through all the layers
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        // iterate through all the neurons in each layer
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            // iterate through all the neurons
            for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                // iterate through all the weights and change them
                layers[layerIndex].neurons[neuronIndex].velocity[weightIndex] *= beta;
                layers[layerIndex].neurons[neuronIndex].velocity[weightIndex] += (1.0 - beta) * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex];
                layers[layerIndex].neurons[neuronIndex].weights[weightIndex] -= learningRate * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] / sqrt(layers[layerIndex].neurons[neuronIndex].velocity[weightIndex] + std::numeric_limits<float>::min());
                layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] = 0;
            }
            // change the bias by the bias gradients
            layers[layerIndex].neurons[neuronIndex].biasVelocity *= beta;
            layers[layerIndex].neurons[neuronIndex].biasVelocity += (1.0 - beta) * layers[layerIndex].neurons[neuronIndex].biasGradient * layers[layerIndex].neurons[neuronIndex].biasGradient;
            layers[layerIndex].neurons[neuronIndex].bias -= learningRate * layers[layerIndex].neurons[neuronIndex].biasGradient / sqrt(layers[layerIndex].neurons[neuronIndex].biasVelocity + std::numeric_limits<float>::min());
            layers[layerIndex].neurons[neuronIndex].biasGradient = 0;
        }
    }
}

void network::ADAM(float learningRate, float beta, float beta2) {
    // iterate through all the layers
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        // iterate through all the neurons in each layer
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].neurons.size(); neuronIndex++) {
            // iterate through all the neurons
            for (int weightIndex = 0; weightIndex < layers[layerIndex].neurons[neuronIndex].weights.size(); weightIndex++) {
                // iterate through all the weights and change them

                // velocity
                layers[layerIndex].neurons[neuronIndex].velocity[weightIndex] *= beta;
                layers[layerIndex].neurons[neuronIndex].velocity[weightIndex] += (1.0 - beta) * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex];

                // momentum
                layers[layerIndex].neurons[neuronIndex].momentum[weightIndex] *= beta2;
                layers[layerIndex].neurons[neuronIndex].momentum[weightIndex] += (1.0 - beta2) * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex];

                // apply gradients
                layers[layerIndex].neurons[neuronIndex].weights[weightIndex] -= learningRate * layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] / sqrt(layers[layerIndex].neurons[neuronIndex].momentum[weightIndex] + std::numeric_limits<float>::min());
                layers[layerIndex].neurons[neuronIndex].weightGradients[weightIndex] = 0;
            }
            // change the bias by the bias gradients

            // velocity
            layers[layerIndex].neurons[neuronIndex].biasVelocity *= beta;
            layers[layerIndex].neurons[neuronIndex].biasVelocity += (1.0 - beta) * layers[layerIndex].neurons[neuronIndex].biasGradient;

            // momentum
            layers[layerIndex].neurons[neuronIndex].biasMomentum *= beta2;
            layers[layerIndex].neurons[neuronIndex].biasMomentum += (1.0 - beta2) * layers[layerIndex].neurons[neuronIndex].biasGradient * layers[layerIndex].neurons[neuronIndex].biasGradient;

            // apply gradients
            layers[layerIndex].neurons[neuronIndex].bias -= learningRate * layers[layerIndex].neurons[neuronIndex].biasGradient / sqrt(layers[layerIndex].neurons[neuronIndex].biasMomentum + std::numeric_limits<float>::min());
            layers[layerIndex].neurons[neuronIndex].biasGradient = 0;
        }
    }
}

void network::initialize(std::default_random_engine& randomEngine, float variance) {
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        layers[layerIndex].initialize(randomEngine, variance);
    }
}

void network::initializeXavier(std::default_random_engine& randomEngine) {
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        layers[layerIndex].initialize(randomEngine, sqrt(6.0f / (layers[layerIndex].neurons[0].weights.size() + layers[layerIndex].neurons.size())));
    }
}

void network::initializeXavier(std::default_random_engine& randomEngine, int index) {
    layers[index].initialize(randomEngine, sqrt(6.0f / (layers[index].neurons[0].weights.size() + layers[index].neurons.size())));
}

void network::initializeHE(std::default_random_engine& randomEngine) {
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        layers[layerIndex].initialize(randomEngine, sqrt(2.0f / layers[layerIndex].neurons[0].weights.size()));
    }
}

void network::initializeHE(std::default_random_engine& randomEngine, int index) {
    layers[index].initialize(randomEngine, sqrt(2.0f / layers[index].neurons[0].weights.size()));
}

void network::mutate(std::default_random_engine& randomEngine, std::normal_distribution<float>& normalDistro) {
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        layers[layerIndex].mutate(randomEngine, normalDistro);
    }
}

void network::print() {
    for (int i = 0; i < layers.size(); i++) {
        std::cout << "Layer " << i << "\n\n";
        layers[i].print();
        std::cout << '\n';
    }
}

void network::printToFile(std::ofstream& file) {
    for (int i = 0; i < layers.size(); i++) {
        file << "Layer " << i << "\n\n";
        layers[i].printToFile(file);
        file << '\n';
    }
}

void readNetworkFromFile(std::string fileName, network* output) {
    std::fstream inputFile(fileName);

    std::string fileLine = "";

    int layerIndex = -1;
    int neuronIndex = 0;

    while (std::getline(inputFile, fileLine)) {
        // when a new layer is found, reset the neuron index and add a layer
        if (fileLine.find("Layer") == 0) {
            layerIndex++;
            output->allocateLayers(layerIndex + 1);
            neuronIndex = 0;
        }

        if (fileLine.find("Neuron") != 0) {
            continue;
        }
        // add a neuron to the outer layer
        output->layers[layerIndex].allocateNeurons(neuronIndex + 1);
        int weightIndex = 0;
        bool weightMode = false;
        bool biasMode = false;
        // when a neuron is found read all of its data
        while (std::getline(inputFile, fileLine)) {
            if (fileLine.find("Bias:") == 0) {
                biasMode = true;
                weightMode = false;
                continue;
            }
            // exit once the bias is found as that is the last part of a neurons data
            if (biasMode) {
                output->layers[layerIndex].neurons[neuronIndex].bias = std::stof(fileLine);
                neuronIndex++;
                break;
            }

            if (fileLine.find("Weights:") == 0) {
                weightMode = true;
                continue;
            }
            if (weightMode) {
                output->layers[layerIndex].neurons[neuronIndex].allocateWeights(weightIndex + 1);
                output->layers[layerIndex].neurons[neuronIndex].weights[weightIndex++] = std::stof(fileLine);
            }
        }
    }
    // close the file
    inputFile.close();
}

void writeNetworkToFile(std::string fileName, network input) {
    std::ofstream file(fileName);
    // print out the network layer sizes
    unsigned long totalWeights = 0;
    unsigned long totalBiases = 0;
    for (int layerIndex = 0; layerIndex < input.layers.size(); layerIndex++) {
        file << "//Layer " << layerIndex << '\n';
        file << "// Neuron Count " << input.layers[layerIndex].neurons.size() << '\n';
        totalBiases += input.layers[layerIndex].neurons.size();
        if (layerIndex == 0) {
            totalWeights += input.layers[layerIndex].neurons[0].weights.size() * input.layers[layerIndex].neurons.size();
        } else {
            totalWeights += input.layers[layerIndex - 1].neurons.size() * input.layers[layerIndex].neurons.size();
        }
    }
    file << "//Total Weights " << totalWeights << '\n';
    file << "//Total Biases " << totalBiases << '\n';
    file << "//Total Parameters " << totalWeights + totalBiases << "\n\n";
    input.printToFile(file);
    file.close();
}
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "NN.h"

#define generations 1'000'000

std::string outputFileName = "networkWeights.txt";

std::string trainDataFileName = "MNIST/mnist_train.csv";
std::string testDataFileName = "MNIST/mnist_test.csv";

/// @brief used to exit traning early
std::atomic<bool> stopTraining(false);

/// @brief sets stop training to true when ctrl + c is pressed
/// @param signum
void signalHandler(int signum) {
    stopTraining = true;
}

// activation functions
float ReLU(float input) {
    return (input > 0 ? input : 0);
}

float ReLUDerivative(float input) {
    return (input > 0);
}

float tanhDerivative(float input) {
    float t = tanh(input);
    return (1 - t * t);
}

float sigmoid(float input) {
    return (1 / (1 + std::exp(-input)));
}

float sigmoidDerivative(float input) {
    float s = sigmoid(input);
    return (s * (1 - s));
}

// cost functions
/*float cost(float input, float target) {
    if (input <= 0) {
        input = std::numeric_limits<float>::lowest();
    } else if (input > 1) {
        input = 1;
    }
    return (-(target * std::log(input) + (1 - target) * std::log(1 - input)));
}

float costDerivative(float input, float target) {
    return ((input - target) / (input * (1 - input)));
}*/

float cost(float input, float target) {
    float dif = input - target;
    return (0.5 * dif * dif);
}

float costDerivative(float input, float target) {
    return (input - target);
}

// simplified version of the output layer derivative
float outputLayerDerivative(float input, float target) {
    return (input - target);
}

/// @brief reads a .CSV file int a 2d vector of floats
/// @param fileName the file to read from
/// @param outputs pointer to the 2d array output
void readCSV(std::string fileName, std::vector<std::vector<float>>* outputs) {
    std::ifstream inputFile(fileName);
    std::string data = "";
    char c = 0;
    char lastc = 0;
    // read the file one byte at a time
    std::vector<float> row;
    while (inputFile.get(c)) {
        // only add numerical characters to the data
        if (c >= '0' && c <= '9' || c == '.' || c == '-') {
            data += c;
            lastc = c;
            continue;
        }
        // dont try to process the data if it is empty
        if (data.size() == 0) {
            continue;
        }
        // if the character was a comma or a newline without a comma prior to it
        // then the data string ended so covert it to a float and add it to the row
        if (c == ',' || c == '\n' && lastc != ',') {
            row.push_back(std::stof(data));
            data = "";
        }
        // if there was a newline and the last row ended
        // add the row the output
        if (c == '\n' && lastc != ',') {
            // dont add any empty rows
            if (row.size() != 0) {
                outputs->push_back(row);
                row.clear();
            }
        }
        lastc = c;
    }
    // add the data to the row if the data is not empty
    if (data.size() != 0) {
        row.push_back(std::stof(data));
    }
    // add the final row if it is not empty
    if (row.size() != 0) {
        outputs->push_back(row);
        row.clear();
    }
    inputFile.close();
}

void printProgressBar(float progress) {
    std::cout << '[';
    for (int i = 0; i < 100; i++) {
        std::cout << ((i < (int)progress) ? "#" : " ");
    }
    std::cout << "] " << std::setprecision(10) << progress << "%\r" << std::flush;
    for (int i = 0; i < 120; i++) {
        std::cout << ' ';
    }
    std::cout << std::setprecision(6) << '\r';
}

void setColor(int textColor = -1, bool textBright = 0, int backgroundColor = -1, bool backgroundBright = 0) {
    if (textColor == -1) {
        std::cout << "\x1b[0m";
        return;
    } else if (backgroundColor == -1) {
        std::cout << "\x1b[49m";
    }
    int colors[] = {30, 31, 33, 32, 34, 36, 35, 37};
    std::string escapeSequence = "\033[";
    escapeSequence += std::to_string(colors[textColor] + 60 * textBright);
    escapeSequence += ';';
    escapeSequence += std::to_string(colors[backgroundColor] + 10 + 60 * backgroundBright);
    escapeSequence += 'm';
    std::cout << escapeSequence;
}

void printDigit(std::vector<float>& input) {
    int index = 0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (input[index] < 0.25) {
                setColor(0, 0, 0, 0);
            } else if (input[index] < 0.5) {
                setColor(0, 0, 0, 1);
            } else if (input[index] < 0.75) {
                setColor(0, 0, 7, 0);
            } else {
                setColor(0, 0, 7, 1);
            }
            std::cout << ' ';
            index++;
        }
        setColor();
        std::cout << '\n';
    }
}

std::vector<float> digitRand(std::vector<float>& input) {
    std::vector<float> output(784);
    int index = 0;
    int xOffset = 3 - rand() % 6;
    int yOffset = 3 - rand() % 6;
    float theta = float(15 - rand() % 30) / 100.0;
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int xTrans = (x * cos(theta)) - (y * sin(theta)) + xOffset;
            int yTrans = (x * sin(theta)) + (y * cos(theta)) + yOffset;
            // skip out of range values
            if (xTrans >= 0 && xTrans < 28 && yTrans >= 0 && yTrans < 28) {
                int outputIndex = xTrans % 28 + yTrans * 28;
                output[outputIndex] = input[index];
            }
            index++;
        }
    }
    for (index = 0; index < 784; index++) {
        if (rand() % 20 == 0) {
            output[index] += float(50 - rand() % 100) / 100.0;
            if (output[index] < 0) {
                output[index] = 0;
            } else if (output[index] > 1) {
                output[index] = 1;
            }
        }
    }
    return (output);
}

float tanhf(float input) {
    return (tanh(input));
}

/// @brief unzips the dataset files
/// @return false on sucess, true on fail
bool unzipDataSets() {
    std::ifstream fin(testDataFileName);
    if (!fin) {
        std::cout << "Unzipping " << testDataFileName << '\n';
        system("unzip MNIST/mnist_test.csv.zip -d MNIST/");
    }
    fin.close();

    fin.open(trainDataFileName);
    if (!fin) {
        std::cout << "Unzipping " << trainDataFileName << '\n';
        system("unzip MNIST/mnist_train.csv.zip -d MNIST/");
    }
    fin.close();

    bool error = 0;
    // test opeining both files
    fin.open(testDataFileName);
    if (!fin) {
        std::cout << "FAILED TO UNZIP " << testDataFileName;
        error = 1;
    }
    fin.close();
    fin.open(trainDataFileName);
    if (!fin) {
        std::cout << "FAILED TO UNZIP " << trainDataFileName;
        error = 1;
    }
    fin.close();
    return (error);
}

int main() {
    // make the random number generator
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::normal_distribution<float> initDistro(0.0, 1);

    if (unzipDataSets()) {
        std::cout << "Maybe try installing unzip?\n";
        return (1);
    }

    // disable scientific notation print out
    std::cout << std::fixed;

    // construct the neural network
    network networkMain;
    // index 0 is the input layer, which is not stored anywhere, it is simply the inputs
    // but it is needed as the first layers neurons must have the same number of weights as inputs
    int layerSizes[] = {784, 200, 100, 10};
    networkMain.allocateNetwork(layerSizes, sizeof(layerSizes) / sizeof(layerSizes[0]) - 1);

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;

    // ask the user what they want to do
    std::cout << "Train Network(t) / Run Network(r)\n";
    std::string input = "";
    while (input != "t" && input != "r") {
        std::cin >> input;
    }

    /**************************
    ***** Run The Network *****
    **************************/

    if (input == "r") {
        // read the data from the testing set
        std::cout << "READING DATA\n";
        readCSV("MNIST/mnist_test.csv", &inputs);
        std::cout << "PROCESSING DATA\n";
        //  seperate the inputs and outputs
        for (long i = 0; i < inputs.size(); i++) {
            std::vector<float> row(10);
            for (int j = 0; j < 10; j++) {
                if (j == inputs[i][0]) {
                    row[j] = 1;
                } else {
                    row[j] = 0;
                }
            }
            outputs.push_back(row);
            // remove the output from the inputs array
            inputs[i].erase(inputs[i].begin());
            for (int j = 0; j < inputs[i].size(); j++) {
                inputs[i][j] /= 255.0;
            }
        }

        // run the data through the network
        std::cout << std::setprecision(3);
        readNetworkFromFile("networkWeights.txt", &networkMain);
        long errors = 0;
        for (int i = 0; i < inputs.size(); i++) {
            networkMain.calculate(inputs[i].data(), tanhf, sigmoid);
            // find the predicted number
            float max = 0;
            int maxIndex = 0;
            for (int j = 0; j < networkMain.layers[networkMain.layers.size() - 1].output.size(); j++) {
                if (networkMain.layers[networkMain.layers.size() - 1].output[j] > max) {
                    max = networkMain.layers[networkMain.layers.size() - 1].output[j];
                    maxIndex = j;
                }
            }
            errors += (outputs[i][maxIndex] != 1);
            printDigit(inputs[i]);
            if (outputs[i][maxIndex] == 1) {
                setColor(3, 0);
            } else {
                setColor(1, 0);
            }
            std::cout << maxIndex;
            setColor();
            std::cout << "\n\n";
        }
        std::cout << "Error Rate: " << (float)errors / (float)inputs.size() * 100.0 << '\n';
        return (0);
    }

    /*******************
    ***** Training *****
    *******************/

    input = "";
    std::cout << "Read network from file(f) / Randomly initialize network(r)\n";
    while (input != "f" && input != "r") {
        std::cin >> input;
    }

    if (input == "f") {
        readNetworkFromFile(outputFileName, &networkMain);
    } else {
        networkMain.initialize(rng, initDistro);
    }

    std::cout << "READING DATA SET\n";
    readCSV("MNIST/mnist_train.csv", &inputs);
    std::cout << "PROCESSING DATA SET\n";
    //  seperate the inputs and outputs
    for (long i = 0; i < inputs.size(); i++) {
        std::vector<float> row(10);
        for (int j = 0; j < 10; j++) {
            if (j == inputs[i][0]) {
                row[j] = 1;
            } else {
                row[j] = 0;
            }
        }
        outputs.push_back(row);
        // remove the output from the inputs array
        inputs[i].erase(inputs[i].begin());
        // normalize the inputs
        for (int j = 0; j < inputs[i].size(); j++) {
            inputs[i][j] /= 255.0;
        }
    }

    std::cout << "TRAINING\n";

    // train the network
    float learingRate = 0.002;
    auto trainingStart = std::chrono::high_resolution_clock::now();
    signal(SIGINT, signalHandler);
    while (!stopTraining) {
        float errorSum = 0;
        long errors = 0;
        bool alreadyCountedError = 0;
        for (int i = 0; i < inputs.size(); i++) {
            networkMain.backPropagation(digitRand(inputs[i]).data(), outputs[i].data(), tanhf, tanhDerivative, sigmoid, sigmoidDerivative, cost, costDerivative);
            float max = 0;
            int maxIndex = 0;
            for (int j = 0; j < outputs[0].size(); j++) {
                if (networkMain.layers[networkMain.layers.size() - 1].output[j] > max) {
                    max = networkMain.layers[networkMain.layers.size() - 1].output[j];
                    maxIndex = j;
                }
            }
            if (outputs[i][maxIndex] != 1) {
                errors++;
            }
            if (i % 50 == 0) {
                networkMain.RMSProp(learingRate);
            }
        }
        networkMain.RMSProp(learingRate);
        float errorRate = (float)errors / (float)inputs.size() * 100.0;
        // clear the terminal
        std::cout << "\x1b[2J\x1b[1;1H" << std::flush << '\n';
        std::cout << "Error Rate: " << errorRate << '\n';
        if (errorRate <= 0.1) {
            break;
        }
    }

    auto trainingStop = std::chrono::high_resolution_clock::now();
    auto trainingDuration = std::chrono::duration_cast<std::chrono::microseconds>(trainingStop - trainingStart);
    std::cout << "Training Done, Took: " << float(trainingDuration.count() / 1'000'000.0) << " seconds" << '\n';

    std::cout << "Calculating test dataset error rate\n";
    inputs.clear();
    outputs.clear();
    readCSV("MNIST/mnist_test.csv", &inputs);
    //  seperate the inputs and outputs
    for (long i = 0; i < inputs.size(); i++) {
        std::vector<float> row(10);
        for (int j = 0; j < 10; j++) {
            if (j == inputs[i][0]) {
                row[j] = 1;
            } else {
                row[j] = 0;
            }
        }
        outputs.push_back(row);
        // remove the output from the inputs array
        inputs[i].erase(inputs[i].begin());
        for (int j = 0; j < inputs[i].size(); j++) {
            inputs[i][j] /= 255.0;
        }
    }

    // run the data through the network
    readNetworkFromFile("networkWeights.txt", &networkMain);
    long errors = 0;
    for (int i = 0; i < inputs.size(); i++) {
        networkMain.calculate(inputs[i].data(), tanhf, sigmoid);
        // find the predicted number
        float max = 0;
        int maxIndex = 0;
        for (int j = 0; j < networkMain.layers[networkMain.layers.size() - 1].output.size(); j++) {
            if (networkMain.layers[networkMain.layers.size() - 1].output[j] > max) {
                max = networkMain.layers[networkMain.layers.size() - 1].output[j];
                maxIndex = j;
            }
        }
        errors += (outputs[i][maxIndex] != 1);
    }
    std::cout << "Error Rate: " << (float)errors / (float)inputs.size() * 100.0 << '\n';

    // ask to store the network in a file
    while (input != "y" && input != "n") {
        std::cout << "Write Network To File Yes(y) / No(n)?\n";
        std::cin >> input;
    }
    if (input == "y") {
        input = "";
        writeNetworkToFile("networkWeights.txt", networkMain);
    }

    return (0);
}
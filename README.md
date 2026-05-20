# Neural Network from Scratch in C++

This project implements a fully connected neural network from scratch in C++ without external machine learning libraries. The implementation includes custom matrix algebra, backpropagation, ReLU activation layers, gradient-based optimization, and training on the XOR classification problem. This project was developed as part of the Object Oriented Scientific Programming with C++ course at TU Delft.

## Features
- Custom matrix class
- Forward and backward propagation
- ReLU activation layers
- Mean squared error (MSE) loss
- Gradient-based optimization
- XOR classification
- Object-oriented neural network architecture

## Network Architecture

The neural network consists of:

- Linear layer
- ReLU activation layer
- Linear output layer

## Training Configuration
The network is trained on the XOR classification problem using:
- learning rate: `0.005`
- hidden layer dimension: `100`
- optimization steps: `100`
- number of samples: `4`

The training loop performs:

- Forward propagation
- Loss computation
- Backward propagation
- Parameter optimization
- Accuracy evaluation

## Requirements
- C++17 compatible compiler
- g++
## Build

Compile the project using:

g++ -std=c++17 -Iinclude src/main.cpp -o neural_network
## Run
### Windows
.\neural_network
### Linux / macOS
./neural_network

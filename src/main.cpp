#include <iostream>
#include "net.hpp"
#include "loss.hpp"

int main()
{

    double learning_rate = 0.005;
    int optimizer_steps = 100;
    int seed = 1;
    int n_samples = 4;
    int in_features = 2;
    int hidden_dim = 100;
    int out_features = 2;
    double* loss_list = new double[optimizer_steps];
    double* accuracy_list = new double[optimizer_steps];

    try {
        Matrix<double> input(n_samples, in_features, {0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0});
        Matrix<double> true_output(n_samples, out_features, {1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0});

        Net<double> net(in_features, hidden_dim, out_features, n_samples, seed);

        for (int i=0; i<optimizer_steps; i++) {
            auto output = net.forward(input);
            auto loss = MSEloss(true_output, output);
            auto gradient = MSEgrad(true_output, output);
            net.backward(gradient);
            net.optimize(learning_rate);
            output = net.forward(input);
            auto accuracy = get_accuracy(true_output, output);

            loss_list[i] = loss;
            accuracy_list[i] = accuracy;
            std::cout << "Iteration = " << i + 1 << ", MSE = " << loss << ", accuracy = " << accuracy << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "Caught an exception: " << e.what() << std::endl;
    }

    delete[] loss_list;
    delete[] accuracy_list;

    loss_list = nullptr;
    accuracy_list = nullptr;

    return 0;
}
#pragma once
#include "linear.hpp"
#include "relu.hpp"

template <typename T>
class Net 
{
// Attributes
private:
    Linear<T> lin1;
    ReLU<T> relu;
    Linear<T> lin2;

public:
    // Constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed)
        : lin1(in_features, hidden_dim, n_samples, seed),
        relu(hidden_dim, hidden_dim, n_samples),
        lin2(hidden_dim, out_features, n_samples, seed)
    {
    }

    // Destructor
    ~Net() {}

    // Forward Function
    Matrix<T> forward(const Matrix<T>& x){
        return lin2.forward(relu.forward(lin1.forward(x))); 
    }

    // Backward Function
    Matrix<T> backward(const Matrix<T>& dy){
        return lin1.backward(relu.backward(lin2.backward(dy)));
    }

    // Optimize Function
    void optimize(T learning_rate){
        lin1.optimize(learning_rate);
        lin2.optimize(learning_rate);
    }
};
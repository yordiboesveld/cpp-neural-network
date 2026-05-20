#pragma once
#include "layer.hpp"
#include <random>

template <typename T>
class Linear: public Layer<T> 
{
private:
    int in_features;
    int out_features;
    int n_samples;
    int seed;
    Matrix<T> cache;
    Matrix<T> bias;
    Matrix<T> weights;
    Matrix<T> bias_gradients;
    Matrix<T> weights_gradients;

public:
    Linear(int in_features, int out_features, int n_samples, int seed) {
        this->in_features = in_features;
        this->out_features = out_features;
        this->n_samples = n_samples;
        this->seed = seed;

        cache = Matrix<T>(n_samples, in_features);
        bias = Matrix<T>(1, out_features);
        weights = Matrix<T>(in_features, out_features);
        bias_gradients = Matrix<T>(1, out_features);
        weights_gradients = Matrix<T>(in_features, out_features);

        std::default_random_engine        generator(seed);
        std::normal_distribution<T>       distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);

        for (int i=0; i<in_features; ++i) {
            for (int j=0; j<out_features; ++j)
                weights[{i,j}] = distribution_normal(generator);
        }

        for (int j=0; j<out_features; ++j)
            bias[{0,j}] = distribution_uniform(generator);
    }

    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        if (x.getRows() != n_samples || x.getCols() != in_features)
            throw std::invalid_argument("Linear::forward: input matrix has wrong shape");
        cache = x;
        Matrix<T> y = x * weights + bias;
        return y;
    }

    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        if (dy.getRows() != n_samples || dy.getCols() != out_features)
            throw std::invalid_argument("Linear::backward: input matrix has wrong shape");
        for (int j = 0; j < out_features; j++) {
            T sum = T(0);
            for (int i = 0; i < dy.getRows(); i++)
                sum += dy[{i, j}];
            bias_gradients[{0, j}] = sum;
        }

        weights_gradients = cache.transpose() * dy;
        Matrix<T> dLdx = dy * weights.transpose();
        return dLdx;
    }

    void optimize(T learning_rate) {
        weights = weights - (weights_gradients * learning_rate);
        bias = bias - (bias_gradients * learning_rate);
    }

    ~Linear() {}
};
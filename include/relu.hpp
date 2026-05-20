#pragma once
#include "layer.hpp"

template <typename T>
class ReLU : public Layer<T> 
{
private:
    int in_features;
    int out_features;
    int n_samples;
    Matrix<T> cache;

public:
    ReLU(int in_features, int out_features, int n_samples) {
        if (in_features != out_features)
            throw std::invalid_argument("ReLU constructor: in_features must equal out_features");
        this->in_features = in_features;
        this->out_features = out_features;
        this-> n_samples = n_samples;
        cache = Matrix<T>(n_samples, in_features);
    }

    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        if (x.getCols() != in_features)
            throw std::invalid_argument("ReLU::forward: input matrix has wrong shape");
        cache = x;
        Matrix<T> y(x.getRows(), x.getCols());
        for (int i=0; i<y.getRows(); i++)
            for (int j=0; j<y.getCols(); j++)
                y[{i,j}] = x[{i,j}] > 0 ? x[{i,j}] : 0;
        return y;
    }

    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        if (dy.getCols() != out_features)
            throw std::invalid_argument("ReLU::backward: input matrix has wrong shape");
        Matrix<T> y(dy.getRows(), dy.getCols());
        for (int i=0; i<y.getRows(); i++)
            for (int j=0; j<y.getCols(); j++)
                y[{i,j}] = cache[{i,j}] > 0 ? dy[{i,j}] : 0;
        return y;
    }

    ~ReLU() {}
};
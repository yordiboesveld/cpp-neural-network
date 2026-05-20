#pragma once
#include "matrix.hpp"

template<typename T>
class Layer
{
public:
    virtual Matrix<T> forward(const Matrix<T>& x) = 0;

    virtual Matrix<T> backward(const Matrix<T>& dy) = 0;

    virtual ~Layer() {}
};
#pragma once
#include "matrix.hpp"

// Function to calculate the loss
template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    if (y_true.getRows() != y_pred.getRows() || y_true.getCols() != y_pred.getCols())
        throw std::invalid_argument("MSEloss: dimension mismatch");

    T result = T(0);
    int rows = y_true.getRows();
    int cols = y_true.getCols();
    int n = rows * cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T diff = y_pred[{i, j}] - y_true[{i, j}];
            result += diff * diff;
        }
    }
    return (n > 0) ? result / static_cast<T>(n) : T(0);
};


// Function to calculate the gradients of the loss
template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    int rows = y_true.getRows();
    int cols = y_true.getCols();

    if (rows != y_pred.getRows() || cols != y_pred.getCols())
        throw std::invalid_argument("MSEgrad: dimension mismatch");

    int n = rows * cols;
    Matrix<T> grad(rows, cols);

    if (n == 0) return grad;

    T scale = static_cast<T>(2) / static_cast<T>(n);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            grad[{i, j}] = scale * (y_pred[{i, j}] - y_true[{i, j}]);

    return grad;
};


// Calculate the argmax 
template <typename T>
Matrix<T> argmax(const Matrix<T>& y) 
{
    Matrix<T> result = Matrix<T>(1,y.getRows());
    for (int i=0; i<y.getRows(); i++) {
        T best_value = y[{i,0}];
        result[{0,i}] = 0;
        for (int j=1; j<y.getCols(); j++) {
            if (y[{i,j}] > best_value) {
                result[{0,i}] = j;
                best_value = y[{i,j}];
            }
        }
    }
    return result;
}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    if (y_pred.getRows() != y_true.getRows() || y_pred.getCols() != y_true.getCols())
        throw std::invalid_argument("get_accuracy: shape of y_pred and y_true mismatch");
    auto argmax_true = argmax(y_true);
    auto argmax_pred = argmax(y_pred);
    T result = T(0);
    for (int j=0; j<argmax_true.getCols(); j++) {
        if (argmax_true[{0,j}] == argmax_pred[{0,j}])
            result += 1;
    }
    return result / static_cast<T>(argmax_true.getCols());
}
#pragma once
#include <initializer_list>
#include <stdexcept>
#include <utility>  
#include <type_traits>  
#include <memory>

template <typename T>
class Matrix
{
    // Attributes
private:
    int nrows;
    int ncols;
    T* data;

public:
    // Default constructor
    Matrix() : nrows(0), ncols(0), data(nullptr) {}

    // Constructor with signature Matrix(int rows, int cols)
    Matrix(int rows, int cols) : nrows(rows), ncols(cols){
        data = new T[nrows * ncols];
        for (int i = 0; i < nrows * ncols; i++){
            data[i] = T(0);
        }
    }

    // Constructor with signature Matrix(int rows, int cols, const std::initializer_list<T>& list)
    Matrix(int rows, int cols, const std::initializer_list<T>& list) : Matrix(rows, cols){
        if ((int)list.size() != nrows * ncols)
            throw std::invalid_argument("Initializer list does not match the dimension rowsxcols");
        
        std::uninitialized_copy(list.begin(), list.end(), data);
    }

    // Copy constructor from another matrix
    Matrix(const Matrix& other) : nrows(other.nrows), ncols(other.ncols){
        int n = nrows * ncols;
        data = (n > 0) ? new T[n] : nullptr;
        for (int i = 0; i < n; i++){
            data[i] = other.data[i];
        }
    }

    // Move constructor from another matrix
    // noexcept: moving a matrix never throws an exception
    Matrix(Matrix&& other) noexcept : nrows(other.nrows), ncols(other.ncols), data(other.data){
        other.nrows = 0;
        other.ncols = 0;
        other.data = nullptr;
    }

    // Destructor 
    ~Matrix(){
        delete[] data;
        data = nullptr;
        nrows = 0;
        ncols = 0;
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other){
        if (this == &other){
            return *this;
        }
        
        delete[] data;

        nrows = other.nrows;
        ncols = other.ncols;
        int n = nrows * ncols;

        data = (n > 0) ? new T[n] : nullptr;
        for (int i =0; i < n; i++){
            data[i] = other.data[i];
        }
        
        return *this;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept{
        if (this == &other){
            return *this;
        }
        
        delete[] data; 
        
        nrows = other.nrows;
        ncols = other.ncols;
        data = other.data;

        other.nrows = 0;
        other.ncols = 0;
        other.data = nullptr;
        
        return *this;
        }

    // Access Operator
    T& operator[](const std::pair<int, int>& ij) {
        int i = ij.first;
        int j = ij.second;

        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            throw std::out_of_range("Requested entry exceeds matrix dimensions");
        }

        return data[ncols * i + j];
    }

    // Constant access Operator
    const T& operator[](const std::pair<int, int>& ij) const{
        int i = ij.first;
        int j = ij.second;

        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            throw std::out_of_range("Requested entry exceeds matrix dimensions");
        }

        return data[ncols * i + j];
    }

    // Arithmetic operator between matrix and scalar; multiplication
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(U x) const{
        using R = typename std::common_type<T, U>::type;

        Matrix<R> result(nrows, ncols);

        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                result[{i,j}] = static_cast<R>((*this)[{i,j}]) * static_cast<R>(x);
        return result;
    }

    // Arithmetic operator between two matrices; multiplication
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> operator*(const Matrix<U>& B) const{
        using R = typename std::common_type<T, U>::type;

        if (ncols != B.getRows())
            throw std::invalid_argument("Matrix multiplication: matrices are not of compatible size");

        Matrix<R> result(nrows, B.getCols());

        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < B.getCols(); j++) {
                R sum = R(0);
                for (int p = 0; p < ncols; p++) {
                    sum += static_cast<R>((*this)[{i, p}])
                        * static_cast<R>(B[{p, j}]);
                }
                result[{i, j}] = sum;
            }
        }
        return result;
    }

    // Arithmetic Operator between two matrices; addition
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> operator+(const Matrix<U>& B) const{
        using R = typename std::common_type<T, U>::type;

        int Arows = nrows;
        int Acols = ncols;
        int Brows = B.getRows();
        int Bcols = B.getCols();

        // Normal case: same shape
        // Special case: B is bias row vector
        bool same_shape = (Arows == Brows && Acols == Bcols);
        bool bias_case = (Brows == 1 && Acols == Bcols);

        if (!same_shape && !bias_case){
            throw std::invalid_argument("Matrix addition: matrices are not of compatible size");
        }

        Matrix<R> result(Arows, Acols);

        if (same_shape){
            for (int i = 0; i < Arows; i++){
                for (int j = 0; j < Acols; j++){
                    result[{i, j}] =
                    static_cast<R>((*this)[{i, j}]) + static_cast<R>(B[{i, j}]);
                }
            }
        } else { // bias case (B has 1 row)
            for (int i = 0; i < Arows; i++){
                for (int j = 0; j < Acols; j++){
                    result[{i, j}] = static_cast<R>((*this)[{i, j}]) + static_cast<R>(B[{0, j}]);
                }
            }
        }

        return result;
    }

    // Arithmetic Operator between two matrices; substraction
    template <typename U>
    Matrix<typename std::common_type<T, U>::type> operator-(const Matrix<U>& B) const{
        using R = typename std::common_type<T, U>::type;

        int Arows = nrows;
        int Acols = ncols;
        int Brows = B.getRows();
        int Bcols = B.getCols();

        // Normal case: same shape
        // Special case: B is bias row vector
        bool same_shape = (Arows == Brows && Acols == Bcols);
        bool bias_case = (Brows == 1 && Acols == Bcols);

        if (!same_shape && !bias_case){
            throw std::invalid_argument("Matrix substraction: matrices are not of compatible size");
        }

        Matrix<R> result(Arows, Acols);

        if (same_shape){
            for (int i = 0; i < Arows; i++){
                for (int j = 0; j < Acols; j++){
                    result[{i, j}] =
                    static_cast<R>((*this)[{i, j}]) - static_cast<R>(B[{i, j}]);
                }
            }
        } else { // bias case (B has 1 row)
            for (int i = 0; i < Arows; i++){
                for (int j = 0; j < Acols; j++){
                    result[{i, j}] = static_cast<R>((*this)[{i, j}]) - static_cast<R>(B[{0, j}]);
                }
            }
        }

        return result;
    }

    // Transpose method
    Matrix transpose() const{
        Matrix<T> result(ncols, nrows);

        for (int i = 0; i < nrows; i++){
            for (int j = 0; j < ncols; j++){
                result[{j, i}] = (*this)[{i, j}];
            }
        }

        return result;
    }

    // Functions
    // getRows
    int getRows() const{
        return nrows;
    }

    // getCols
    int getCols() const{
        return ncols;
    }
 
};
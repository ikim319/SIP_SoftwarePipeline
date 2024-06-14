/* kalman.i */
%module kalman

%{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "kalman.h"
#include "eigen-3.4.0/Eigen/Dense"
#include <numpy/arrayobject.h>

// Helper functions
Eigen::MatrixXd numpy_to_eigen(PyObject* o) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(o);
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected a 2D array");
        return Eigen::MatrixXd();
    }
    int rows = PyArray_DIM(arr, 0);
    int cols = PyArray_DIM(arr, 1);
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = *static_cast<double*>(PyArray_GETPTR2(arr, i, j));
        }
    }
    return mat;
}

Eigen::VectorXd numpy_to_eigen_vec(PyObject* o) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(o);
    if (PyArray_NDIM(arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected a 1D array");
        return Eigen::VectorXd();
    }
    int size = PyArray_DIM(arr, 0);
    Eigen::VectorXd vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = *static_cast<double*>(PyArray_GETPTR1(arr, i));
    }
    return vec;
}
%}

%include "numpy.i"

// Typemaps for converting NumPy arrays to Eigen matrices and vectors
%typemap(in) (const Eigen::MatrixXd &INPUT) (PyObject *o) {
    $1 = numpy_to_eigen(o);
}

%typemap(in) (const Eigen::VectorXd &INPUT) (PyObject *o) {
    $1 = numpy_to_eigen_vec(o);
}

// Typemaps for converting Eigen matrices and vectors to NumPy arrays
%typemap(out) Eigen::MatrixXd {
    npy_intp dims[2] = { $1.rows(), $1.cols() };
    PyObject* obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!obj) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory for output array");
        SWIG_fail;
    }
    for (int i = 0; i < $1.rows(); ++i) {
        for (int j = 0; j < $1.cols(); ++j) {
            *static_cast<double*>(PyArray_GETPTR2((PyArrayObject*)obj, i, j)) = $1(i, j);
        }
    }
    $result = obj;
}

%typemap(out) Eigen::VectorXd {
    npy_intp dims[1] = { $1.size() };
    PyObject* obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!obj) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate memory for output array");
        SWIG_fail;
    }
    for (int i = 0; i < $1.size(); ++i) {
        *static_cast<double*>(PyArray_GETPTR1((PyArrayObject*)obj, i)) = $1(i);
    }
    $result = obj;
}

// Include kalman.h without redefining the class
%include "kalman.h"

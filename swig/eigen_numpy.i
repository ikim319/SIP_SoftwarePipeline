// eigen_numpy.i

%module(directors="1") eigen_numpy

%{
#include <Eigen/Core>
#include <numpy/arrayobject.h>

// Initialize the NumPy C API
void init_numpy() {
  import_array();
}

// A templated class to map numpy arrays to Eigen matrices
template<typename MatType>
MatType numpy_to_eigen(PyObject* obj) {
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj);
  double* data = static_cast<double*>(PyArray_DATA(array));
  int rows = PyArray_DIMS(array)[0];
  int cols = PyArray_DIMS(array)[1];
  return Eigen::Map<MatType>(data, rows, cols);
}

%}

// Include Eigen and NumPy
%include "Eigen/Core"
%include "numpy.i"

// Use Eigen and NumPy
%init %{
import_array();
%}

// Typemap for converting numpy arrays to Eigen matrices
%typemap(in) (Eigen::MatrixXd) {
  $1 = numpy_to_eigen<Eigen::MatrixXd>($input);
}

%typemap(in) (const Eigen::MatrixXd &) {
  $1 = numpy_to_eigen<Eigen::MatrixXd>($input);
}

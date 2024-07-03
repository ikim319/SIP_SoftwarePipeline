// kalman.i

%module kalman

%{
#include "kalman.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Eigen/Dense>
%}

/* Include the SWIG interface for NumPy */
%include "numpy.i"

/* Include the SWIG interface for Eigen to NumPy conversion */
%include "eigen_numpy.i"

/* Initialize NumPy support */
%init %{
import_array();
%}

/* Include the header file */
%include "kalman.hpp"

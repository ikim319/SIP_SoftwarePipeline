%module kalman

%{
#include "kalman.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
%}

/* Include the SWIG interface for NumPy */
%include "numpy.i"

/* Initialize NumPy support */
%init %{
import_array();
%}

/* Define the custom macros */
%{
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>

#if defined(NPY_API_VERSION) && NPY_API_VERSION >= 0x00000007

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#else

#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_CONTIGUOUS
#endif

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

#ifndef PyArray_EnableFlags
#define PyArray_EnableFlags(arr, flags) ((arr)->flags |= (flags))
#endif

#ifndef PyArray_CLEARFLAGS
#define PyArray_CLEARFLAGS(arr, flags) ((arr)->flags &= ~(flags))
#endif

#ifndef PyArray_ENABLEFLAGS
#define PyArray_ENABLEFLAGS(arr, flags) ((arr)->flags |= (flags))
#endif

#endif
%}

/* Define the typemaps for NumPy arrays */
%typemap(in) (double* input_array, int input_size) {
    if (!PyArray_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *) $input;
    if (PyArray_NDIM(arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        return NULL;
    }
    if (PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Expected an array of type double");
        return NULL;
    }
    $1 = (double*) PyArray_DATA(arr);
    $2 = PyArray_DIM(arr, 0);
}

%typemap(argout) (double* output_array, int output_size) {
    npy_intp dims[1] = { $2 };
    PyArrayObject *arr = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for output array");
        return NULL;
    }
    memcpy(PyArray_DATA(arr), $1, $2 * sizeof(double));
    $result = (PyObject*) arr;
}

/* Include the KalmanFilter class */
%include "kalman.hpp"

/* kalman_basic.i */
%module kalman_basic

%{
    #include "kalman_basic.h"
%}

%include "std_vector.i"

// Define typemaps for std::tuple
%typemap(out) std::tuple<double, double> {
    $result = PyTuple_Pack(2, PyFloat_FromDouble(std::get<0>($1)), PyFloat_FromDouble(std::get<1>($1)));
}

std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn);

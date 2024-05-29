/* kalman_basic.i */
%module kalman_basic

%{
    #include "kalman_basic.h"
%}

%include "std_tuple.i"
%include "std_vector.i"

std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn);

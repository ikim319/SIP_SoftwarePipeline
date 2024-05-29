/* kalman_basic.i */
%module kalman_basic

%{
    #include "kalman_basic.cpp"
%}

%include "std_vector.i"

%{
    // Declaration of functions to be exposed
    std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn);
%}

%include "std_tuple.i"

std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn);

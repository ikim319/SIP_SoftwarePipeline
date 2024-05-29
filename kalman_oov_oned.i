/* kalman_basic.i */
%module kalman_basic

%{
    #include "kalman_oov_oned.cpp"
%}

%include "std_vector.i"

class KalmanFilter {
public:
    KalmanFilter(double pn, double mn, double ee, double iv);
    double update(double measurement);
};

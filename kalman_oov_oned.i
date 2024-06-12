/* kalman_oov_oned.i */
%module kalman_oov_oned

%{
    #include "kalman_oov_oned.h"
%}

%include "std_vector.i"

// Expose the KalmanFilter class to Python
class KalmanFilter {
public:
    KalmanFilter(double pn, double mn, double ee, double iv);
    double update(double measurement);
};
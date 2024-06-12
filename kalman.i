/* kalman.i */
%module kalman

%{
    #include "kalman.h"
    #include "Eigen/Dense"
%}

%include "std_vector.i"
%include "Eigen.i"

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int meas_dim);
    void predict();
    void update(const Eigen::VectorXd &z);
    void setStateTransition(const Eigen::MatrixXd &F);
    void setMeasurementMatrix(const Eigen::MatrixXd &H);
    void setProcessNoiseCovariance(const Eigen::MatrixXd &Q);
    void setMeasurementNoiseCovariance(const Eigen::MatrixXd &R);
    void setInitialState(const Eigen::VectorXd &x0);
    void setInitialCovariance(const Eigen::MatrixXd &P0);
    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;
};

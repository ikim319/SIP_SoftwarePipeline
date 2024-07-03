#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "eigen-3.4.0/Eigen/Dense"

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

private:
    Eigen::VectorXd x_; // State vector
    Eigen::MatrixXd P_; // State covariance matrix
    Eigen::MatrixXd F_; // State transition matrix
    Eigen::MatrixXd H_; // Measurement matrix
    Eigen::MatrixXd R_; // Measurement noise covariance matrix
    Eigen::MatrixXd Q_; // Process noise covariance matrix
    Eigen::MatrixXd I_; // Identity matrix
};

#endif // KALMAN_HPP

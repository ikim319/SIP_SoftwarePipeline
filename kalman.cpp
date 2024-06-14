#include "kalman.h"

KalmanFilter::KalmanFilter(int state_dim, int meas_dim) {
    x_ = Eigen::VectorXd::Zero(state_dim); // State vector
    P_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // State covariance matrix
    F_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // State transition matrix
    H_ = Eigen::MatrixXd::Zero(meas_dim, state_dim); // Measurement matrix
    R_ = Eigen::MatrixXd::Identity(meas_dim, meas_dim); // Measurement noise covariance matrix
    Q_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // Process noise covariance matrix
    I_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // Identity matrix
}

void KalmanFilter::predict() {
    x_ = F_ * x_; // Predicted state estimate
    P_ = F_ * P_ * F_.transpose() + Q_; // Predicted estimate covariance
}

void KalmanFilter::update(const Eigen::VectorXd &z) {
    Eigen::VectorXd y = z - H_ * x_; // Measurement residual
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_; // Measurement residual covariance
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse(); // Kalman gain
    x_ = x_ + K * y; // Updated state estimate
    P_ = (I_ - K * H_) * P_; // Updated estimate covariance
}

void KalmanFilter::setStateTransition(const Eigen::MatrixXd &F) {
    F_ = F;
}

void KalmanFilter::setMeasurementMatrix(const Eigen::MatrixXd &H) {
    H_ = H;
}

void KalmanFilter::setProcessNoiseCovariance(const Eigen::MatrixXd &Q) {
    Q_ = Q;
}

void KalmanFilter::setMeasurementNoiseCovariance(const Eigen::MatrixXd &R) {
    R_ = R;
}

void KalmanFilter::setInitialState(const Eigen::VectorXd &x0) {
    x_ = x0;
}

void KalmanFilter::setInitialCovariance(const Eigen::MatrixXd &P0) {
    P_ = P0;
}

Eigen::VectorXd KalmanFilter::getState() const {
    return x_;
}

Eigen::MatrixXd KalmanFilter::getCovariance() const {
    return P_;
}

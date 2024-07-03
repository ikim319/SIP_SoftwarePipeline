#include "kalman.hpp"
#include <Eigen/Dense>

KalmanFilter::KalmanFilter(int state_dim, int meas_dim) {
    x_ = Eigen::VectorXd::Zero(state_dim);
    P_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
    F_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H_ = Eigen::MatrixXd::Zero(meas_dim, state_dim);
    R_ = Eigen::MatrixXd::Identity(meas_dim, meas_dim);
    Q_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
    I_ = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::VectorXd &z) {
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;
    P_ = (I_ - K * H_) * P_;
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

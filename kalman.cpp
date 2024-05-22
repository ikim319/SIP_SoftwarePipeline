#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <vector>
#include <chrono>

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int meas_dim) {
        x_ = Eigen::VectorXd::Zero(state_dim); // State vector
        P_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // State covariance matrix
        F_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // State transition matrix
        H_ = Eigen::MatrixXd::Zero(meas_dim, state_dim); // Measurement matrix
        R_ = Eigen::MatrixXd::Identity(meas_dim, meas_dim); // Measurement noise covariance matrix
        Q_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // Process noise covariance matrix
        I_ = Eigen::MatrixXd::Identity(state_dim, state_dim); // Identity matrix
    }

    void predict() {
        // Predicted state estimate
        x_ = F_ * x_;
        // Predicted estimate covariance
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Eigen::VectorXd &z) {
        // Measurement residual
        Eigen::VectorXd y = z - H_ * x_;
        // Measurement residual covariance
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
        // Kalman gain
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
        // Updated state estimate
        x_ = x_ + K * y;
        // Updated estimate covariance
        P_ = (I_ - K * H_) * P_;
    }

    void setStateTransition(const Eigen::MatrixXd &F) {
        F_ = F;
    }

    void setMeasurementMatrix(const Eigen::MatrixXd &H) {
        H_ = H;
    }

    void setProcessNoiseCovariance(const Eigen::MatrixXd &Q) {
        Q_ = Q;
    }

    void setMeasurementNoiseCovariance(const Eigen::MatrixXd &R) {
        R_ = R;
    }

    void setInitialState(const Eigen::VectorXd &x0) {
        x_ = x0;
    }

    void setInitialCovariance(const Eigen::MatrixXd &P0) {
        P_ = P0;
    }

    Eigen::VectorXd getState() const {
        return x_;
    }

    Eigen::MatrixXd getCovariance() const {
        return P_;
    }

private:
    Eigen::VectorXd x_; // State vector
    Eigen::MatrixXd P_; // State covariance matrix
    Eigen::MatrixXd F_; // State transition matrix
    Eigen::MatrixXd H_; // Measurement matrix
    Eigen::MatrixXd R_; // Measurement noise covariance matrix
    Eigen::MatrixXd Q_; // Process noise covariance matrix
    Eigen::MatrixXd I_; // Identity matrix
};

int main() {
    // Dimensions
    int state_dim = 2; // For example: position and velocity
    int meas_dim = 1; // For example: position measurement

    // Create Kalman Filter instance
    KalmanFilter kf(state_dim, meas_dim);

    // Define state transition matrix (F)
    Eigen::MatrixXd F(state_dim, state_dim);
    F << 1, 1,
         0, 1;
    kf.setStateTransition(F);

    // Define measurement matrix (H)
    Eigen::MatrixXd H(meas_dim, state_dim);
    H << 1, 0;
    kf.setMeasurementMatrix(H);

    // Define process noise covariance matrix (Q)
    Eigen::MatrixXd Q(state_dim, state_dim);
    Q << 1, 0,
         0, 1;
    kf.setProcessNoiseCovariance(Q);

    // Define measurement noise covariance matrix (R)
    Eigen::MatrixXd R(meas_dim, meas_dim);
    R << 1;
    kf.setMeasurementNoiseCovariance(R);

    // Initial state estimate
    Eigen::VectorXd x0(state_dim);
    x0 << 0, 0;
    kf.setInitialState(x0);

    // Initial covariance estimate
    Eigen::MatrixXd P0(state_dim, state_dim);
    P0 << 1, 0,
          0, 1;
    kf.setInitialCovariance(P0);

    // Simulated measurements
    std::vector<Eigen::VectorXd> measurements;
    measurements.push_back((Eigen::VectorXd(1) << 1).finished());
    measurements.push_back((Eigen::VectorXd(1) << 2).finished());
    measurements.push_back((Eigen::VectorXd(1) << 3).finished());

    // Kalman filter loop (benchmarking)
    const int num_runs = 1000;
    std::vector<double> times;

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        for (const auto &z : measurements) {
            kf.predict();
            kf.update(z);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    double sum = 0;
    for (const auto &time : times) {
        sum += time;
    }

    double avg_time = sum * 1000000 / num_runs;

    std::cout << "Average time for " << num_runs << " runs: " << avg_time << " microseconds" << std::endl;

    return 0;
}

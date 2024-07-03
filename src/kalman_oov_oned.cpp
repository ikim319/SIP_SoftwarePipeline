#include <iostream>
#include <vector>
#include <chrono>
#include "kalman_oov_oned.h"

// Constructor for KalmanFilter
KalmanFilter::KalmanFilter(double pn, double mn, double ee, double iv)
    : q(pn), r(mn), p(ee), x(iv) {}

// Update method for KalmanFilter
double KalmanFilter::update(double measurement) {
    // Prediction update
    p = p + q; // Covariance extrapolation equation

    // Measurement update
    double k = p / (p + r); // Kalman gain
    x = x + k * (measurement - x); // Status update
    p = (1 - k) * p; // Covariance update

    return x;
}

int main() {
    double pn = 1e-5;
    double mn = 1e-5;
    double ee = 1;
    double iv = 0;

    // Example measurements
    std::vector<double> measurements = {1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0};

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    KalmanFilter kf(pn, mn, ee, iv);
    std::cout << "Filtered Values:" << std::endl;
    for(double m : measurements) {
        double result = kf.update(m);
        std::cout << "Measurement: " << m << "; Filtered: " << result << std::endl;
    }

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    // Elapsed time calculation
    std::chrono::duration<double> elapsed = (end - start) * 1000000;
    std::cout << "Time taken: " << elapsed.count() << " microseconds" << std::endl;

    return 0;
}
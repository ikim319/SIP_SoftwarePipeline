#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>
#include "kalman_oov_oned.h"
//1-D kalman filter
class KalmanFilter {
public:
    //constructor
    KalmanFilter(double pn, double mn, double ee, double iv) {
        q = pn; //process noise covariance
        r = mn; //measurement noise covariance
        p = ee; //estimated error covariance
        x = iv; //initial value
    }

    double update(double measurement) {
        //prediction update
        p = p + q; //covariance extrapolation equation

        //measurement update
        double k = p / (p + r); //kalman gain
        x = x + k * (measurement - x); //status update
        p = (1 - k) * p; //covariance update

        return x;
    }
    
private:
    double q;
    double r;
    double p;
    double x; //curr val
};

int main() {
    double pn = 1e-5;
    double mn = 1e-5;
    double ee = 1;
    double iv = 0;

    //example measurements
    std::vector<double> measurements = {1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0};

    //start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    KalmanFilter kf(pn, mn, ee, iv);
    std::cout << "Filtered Values:" << std::endl;
    for(double m : measurements) {
        double result = kf.update(m);
        std::cout << "Measurement: " << m << "; Filtered: " << result << std::endl;
    }

    //end time measurement
    auto end = std::chrono::high_resolution_clock::now();
    //elapsed time calculation
    std::chrono::duration<double> elapsed = (end - start) * 1000000;
    std::cout << "Time taken: " << elapsed.count() << " microseconds" << std::endl;

    return 0;
};
#include <iostream>
#include <tuple>
#include <vector>
//1-D kalman filter functional version
std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn) {
    double q = pn; //process noise covariance
    double r = mn; //measurement noise covariance
    
    //prediction update
    p = p + q; //covariance extrapolation equation

    //measurement update
    double k = p / (p + r); //kalman gain
    x = x + k * (measurement - x); //status update
    p = (1 - k) * p; //covariance update

    return std::make_tuple(p, x); //updated (covariance, state)
}

int main() {
    double p = 1; //estimated error
    double x = 0; //initial value
    double pn = 1e-5;
    double mn = 1e-5;

    //example measurements
    std::vector<double> measurements = {1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0};

    for (double m : measurements) {
        std::tie(p, x) = kalman_update(p, x, m, pn, mn);
        std::cout << "Measurement: " << m << "; Filtered: " << x << std::endl;
    }

    return 0;
};
// kalman.h
#ifndef KALMAN_H
#define KALMAN_H

#include <vector>
#include <tuple>
#include <chrono>

class KalmanFilter {
public:
    KalmanFilter(double pn, double mn, double ee, double iv);
    double update(double measurement);
    
private:
    double q;
    double r;
    double p;
    double x;
};

#endif // KALMAN_H

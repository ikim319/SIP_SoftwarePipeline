#ifndef KALMAN_BASIC_H
#define KALMAN_BASIC_H

#include <tuple>
#include <vector>
#include <chrono>

// Function declaration for the 1-D Kalman filter
std::tuple<double, double> kalman_update(double p, double x, double measurement, double pn, double mn);

#endif // KALMAN_BASIC_H

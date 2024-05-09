#include <iostream>
using namespace std;
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

    KalmanFilter kf(pn, mn, ee, iv);
    double measurement = 0;
    cout << "Filtered Values:" << endl;
    for(int i = 0; i < 10; i++) {
        measurement++; //measurement logic goes here
        double result = kf.update(measurement);
        cout << "Measurement: " << measurement << "; Filtered: " << result << endl;
    }

    return 0;
};
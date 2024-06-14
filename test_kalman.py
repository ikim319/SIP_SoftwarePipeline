import kalman
import numpy as np

state_dim = 2
meas_dim = 1

kf = kalman.KalmanFilter(state_dim, meas_dim)

F = np.array([[1, 1], [0, 1]], dtype=np.float64)
kf.setStateTransition(F)

H = np.array([[1, 0]], dtype=np.float64)
kf.setMeasurementMatrix(H)

Q = np.array([[1, 0], [0, 1]], dtype=np.float64)
kf.setProcessNoiseCovariance(Q)

R = np.array([[1]], dtype=np.float64)
kf.setMeasurementNoiseCovariance(R)

x0 = np.array([0, 0], dtype=np.float64)
kf.setInitialState(x0)

P0 = np.array([[1, 0], [0, 1]], dtype=np.float64)
kf.setInitialCovariance(P0)

measurements = [np.array([1], dtype=np.float64), np.array([2], dtype=np.float64), np.array([3], dtype=np.float64)]

for z in measurements:
    kf.predict()
    kf.update(z)
    print("Filtered State:", kf.getState())

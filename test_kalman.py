import kalman
import numpy as np

# Define dimensions
state_dim = 4
meas_dim = 2

# Create a Kalman Filter object
kf = kalman.KalmanFilter(state_dim, meas_dim)

# Set initial state
initial_state = np.array([0.0, 0.0, 0.0, 0.0])
kf.setInitialState(initial_state)

# Set state transition matrix
F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
kf.setStateTransition(F)

# Set measurement matrix
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
kf.setMeasurementMatrix(H)

# Set process noise covariance matrix
Q = np.eye(state_dim) * 0.01
kf.setProcessNoiseCovariance(Q)

# Set measurement noise covariance matrix
R = np.eye(meas_dim) * 0.1
kf.setMeasurementNoiseCovariance(R)

# Set initial covariance matrix
P0 = np.eye(state_dim)
kf.setInitialCovariance(P0)

# Perform a prediction step
kf.predict()

# Create a measurement vector
z = np.array([1.0, 1.0])
kf.update(z)

# Get the updated state and covariance
updated_state = kf.getState()
updated_covariance = kf.getCovariance()

print("Updated state:")
print(updated_state)
print("Updated covariance:")
print(updated_covariance)

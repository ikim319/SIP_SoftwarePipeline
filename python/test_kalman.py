import numpy as np
import kalman

# Define the initial state and covariance
initial_state = np.array([0, 0])
initial_covariance = np.array([[1, 0], [0, 1]])

# Define the state transition matrix
state_transition = np.array([[1, 1], [0, 1]])

# Define the measurement matrix
measurement_matrix = np.array([[1, 0]])

# Define the process noise covariance matrix
process_noise_covariance = np.array([[1, 0], [0, 1]])

# Define the measurement noise covariance matrix
measurement_noise_covariance = np.array([[1]])

# Create a KalmanFilter object
kf = kalman.KalmanFilter(2, 1)

# Set the parameters of the Kalman filter
kf.setInitialState(initial_state)
kf.setInitialCovariance(initial_covariance)
kf.setStateTransition(state_transition)
kf.setMeasurementMatrix(measurement_matrix)
kf.setProcessNoiseCovariance(process_noise_covariance)
kf.setMeasurementNoiseCovariance(measurement_noise_covariance)

# Define the measurement vector
measurement = np.array([1])

# Predict the next state
kf.predict()

# Update the state with the measurement
kf.update(measurement)

# Get the updated state and covariance
updated_state = kf.getState()
updated_covariance = kf.getCovariance()

print("Updated state:", updated_state)
print("Updated covariance:", updated_covariance)

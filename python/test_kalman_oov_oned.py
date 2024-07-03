import kalman_oov_oned
import time

pn = 1e-5
mn = 1e-5
ee = 1
iv = 0
measurements = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0]

kf = kalman_oov_oned.KalmanFilter(pn, mn, ee, iv)

# Start the timer
start_time = time.time()

for m in measurements:
    result = kf.update(m)
    print(f"Measurement: {m}; Filtered: {result}")

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time) * 1000000  # convert to microseconds

print(f"Time taken: {elapsed_time:.2f} microseconds")
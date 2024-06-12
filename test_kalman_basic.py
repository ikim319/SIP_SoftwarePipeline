import kalman_basic
import time

p = 1
x = 0
pn = 1e-5
mn = 1e-5
measurements = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0]

#start timer
start_time = time.time()

for m in measurements:
    p, x = kalman_basic.kalman_update(p, x, m, pn, mn)
    print(f"Measurement: {m}; Filtered: {x}")

#end timer
end_time = time.time()
elapsed_time = (end_time - start_time) * 1000000 #convert to microseconds
print(f"Time taken: {elapsed_time:.2f} microseconds")
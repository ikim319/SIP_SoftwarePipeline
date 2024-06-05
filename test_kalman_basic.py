import kalman_basic

p = 1
x = 0
pn = 1e-5
mn = 1e-5
measurements = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0]

for m in measurements:
    p, x = kalman_basic.kalman_update(p, x, m, pn, mn)
    print(f"Measurement: {m}; Filtered: {x}")

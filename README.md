# SIP_ResearchProject
Real-Time Software Improvement Project for SIPLab under Dr. Christopher Rozell
- Benchmarking comparison of C++ code to calling SWIG interface from different languages
- To compile and run C++ code:
  1. g++ -std=c++11 kalman_basic.cpp -o kalman_basic
  2. ./kalman_basic
- Kalman Filter
  - algorithm used to estimate the state of a dynamic system from a series of noisy measurements that consists of the following steps:
  - Predict:
    - Estimate the next state based on the current state and system model.
    - Calculate the expected uncertainty of this prediction.
  - Update:
    - Incorporate new measurements to refine the state estimate.
    - Adjust the uncertainty based on the measurement accuracy.
  - Iterate:
    - Repeat the predict and update steps for each new measurement, continuously refining the state estimate and reducing uncertainty.

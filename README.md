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
- Steps for SWIG:
  1. Generate SWIG wrapper: swig -python -c++ kalman.i
  2. Compile wrapper and C++ code: g++ -shared -o _kalman.so kalman_wrap.cxx kalman.cpp -I/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10 -L/Library/Frameworks/Python.framework/Versions/3.10/lib -lpython3.10 -std=c++11
  3. Create Python script that utilizes the module
  4. Run the Python script: python3 test_kalman.py

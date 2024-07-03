find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

set(CMAKE_SWIG_FLAGS "")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/swig)

set(SWIG_MODULE_kalman_EXTRA_DEPS ${CMAKE_SOURCE_DIR}/src/kalman.hpp)

swig_add_library(kalman
    LANGUAGE python
    SOURCES kalman.i kalman.cpp)

swig_link_libraries(kalman ${PYTHON_LIBRARIES})

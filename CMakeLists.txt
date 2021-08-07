cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 3.3 REQUIRED)

add_executable(example
    example.cpp    
    EKF.cpp
)

target_link_libraries(example Eigen3::Eigen)
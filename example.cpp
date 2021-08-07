#include <iostream>
#include "EKF.hpp"
#include <Eigen/Core>

using namespace std;

int main(int argc, char const *argv[])
{
    LinearSensorParam lidar_xy;
    lidar_xy.sensor_dim = 2;
    lidar_xy.sensor_dim = 4;
    lidar_xy.H = Eigen::MatrixXd(2, 4);
    lidar_xy.H(0, 0) = 1;
    lidar_xy.H(1, 1) = 1;
    lidar_xy.V = Eigen::Matrix2d::Identity();

    LinearSensorWrapper lidar_sensor(lidar_xy);

    Eigen::Vector4d x;
    x << 1, 2, 3, 4;

    cout << lidar_sensor.measure(x) << endl;
    // cout << lidar_sensor.jacobian(x) << endl;

    return 0;
}

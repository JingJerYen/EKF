#include "EKF.hpp"

EKF::EKF(std::unique_ptr<MotionBase> motion) : motion_(std::move(motion)),
                                               state_dim(motion_->stateDimension())
{
    I = Eigen::MatrixXd(state_dim, state_dim);
    I.setIdentity();
    Q = motion_->covariance();
}

int EKF::registerSensor(std::unique_ptr<SensorBase> sensor)
{
    int meas_dim = sensor->measureDimension();
    sensor_dim.push_back(meas_dim);
    K.push_back(Eigen::MatrixXd(state_dim, meas_dim));
    V.push_back(sensor->covariance());
    sensors_.emplace_back(std::move(sensor));

    return int(sensors_.size()) - 1;
}

void EKF::propagate()
{
    Eigen::MatrixXd F = motion_->jacobian(x);
    x = motion_->forward(x);
    P = F * P * F.transpose() + Q;
}

void EKF::update(int sensor_id, const Eigen::VectorXd &measurement)
{
    Eigen::MatrixXd H = sensors_[sensor_id]->jacobian(x);
    Eigen::MatrixXd PH_t = P * H.transpose();
    // K[sensor_id] = PH_t * (H * PH_t + V[sensor_id]).inverse();
    K[sensor_id] = PH_t * (H * PH_t + V[sensor_id]);

    x = x + K[sensor_id] * (measurement - sensors_[sensor_id]->measure(x));

    // Joseph form
    Eigen::MatrixXd I_KH = I - K[sensor_id] * H;
    P = I_KH * P * I_KH.transpose() + K[sensor_id] * V[sensor_id] * K[sensor_id].transpose();
}
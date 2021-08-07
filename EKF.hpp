#pragma once
#include <Eigen/Core>
#include <vector>
#include <memory>

bool checkSysmetric(const Eigen::MatrixXd &m);

Eigen::VectorXd sampleFromCovariance(const Eigen::MatrixXd &m);

/**
 * @brief measurement is : z = Hx + noise~N(0, V)
 */
struct LinearSensorParam
{
    int sensor_dim;
    int state_dim;

    // sensor covariance [sensor_dim x sensor_dim]
    Eigen::MatrixXd V;

    // sensor jacobian z = Hx, [sensor_dim x state_dim]
    Eigen::MatrixXd H;
};

class SensorBase
{
public:
    virtual ~SensorBase() {}
    virtual Eigen::VectorXd measure(const Eigen::VectorXd &state) = 0;
    virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd &state) = 0;
    virtual Eigen::MatrixXd covariance() = 0;
    virtual int stateDimension() = 0;
    virtual int measureDimension() = 0;
};

class LinearSensorWrapper : public SensorBase
{
public:
    LinearSensorWrapper(const LinearSensorParam &params) : params_(params){};
    ~LinearSensorWrapper(){};

    Eigen::VectorXd measure(const Eigen::VectorXd &state)
    {
        return params_.H * state + params_.V.diagonal();
        // return params_.H * state + sampleFromCovariance(params_.V);
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd &state)
    {
        return params_.H;
    }

    Eigen::MatrixXd covariance() { return params_.V; }
    int stateDimension() { return params_.state_dim; }
    int measureDimension() { return params_.sensor_dim; }

private:
    LinearSensorParam params_;
};

class MotionBase
{
public:
    virtual ~MotionBase() {}
    virtual Eigen::VectorXd forward(const Eigen::VectorXd &state) = 0;
    virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd &state) = 0;
    virtual Eigen::MatrixXd covariance() = 0;
    virtual int stateDimension() = 0;
};

struct LinearMotionParam
{
    int state_dim;

    // motion intrinsic covariance, [state_dim * state_dim]
    Eigen::MatrixXd Q;

    // propagate matrix or jacobian, x = Fx + Q
    Eigen::MatrixXd F;
};

class LinearMotionWrapper : public MotionBase
{
public:
    LinearMotionWrapper(const LinearMotionParam &params) : params_(params) {}
    ~LinearMotionWrapper() {}

    Eigen::VectorXd forward(const Eigen::VectorXd &state)
    {
        return params_.F * state + params_.Q.diagonal();
        // return params_.F * state + sampleFromCovariance(Q);
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd &state)
    {
        return params_.F;
    }

    Eigen::MatrixXd covariance()
    {
        return params_.Q;
    };

    int stateDimension()
    {
        return params_.state_dim;
    }

private:
    LinearMotionParam params_;
};

/**
 * @brief A stateful object.
 */
class EKF
{
public:
    EKF(std::unique_ptr<MotionBase> motion);
    ~EKF();

    void resetState(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0)
    {
        x = x0;
        P = P0;
    };

    /**
     * @brief add a sensor model
     * @param sensor
     * @return index of sensor (from 0) 
     */
    int registerSensor(std::unique_ptr<SensorBase> sensor);

    void propagate();

    void update(int sensor_id, const Eigen::VectorXd &measurement);

    const Eigen::VectorXd &getCurrentState() { return x; }
    const Eigen::MatrixXd &getCurrentCovar() { return P; }

private:
    std::unique_ptr<MotionBase> motion_;

    std::vector<std::unique_ptr<SensorBase>> sensors_;

    int state_dim;

    std::vector<int> sensor_dim;

    // current state [state_dim x 1]
    Eigen::VectorXd x;

    // current state covariance [state_dim x state_dim]
    Eigen::MatrixXd P;

    // kalman gain, [state_dim x sensor_dim]
    std::vector<Eigen::MatrixXd> K;

    // sensor covariance [sensor_dim x sensor_dim]
    std::vector<Eigen::MatrixXd> V;

    // motion covariance [state_dim x state_dim]
    Eigen::MatrixXd Q;

    // Identity, [state_dim * state_dim]
    Eigen::MatrixXd I;
};
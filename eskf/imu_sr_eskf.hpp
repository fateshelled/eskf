#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

// Square-Root Error-State Kalman Filter (SR-ESKF) implementation for a 15-state IMU-driven model.
class IMU_SR_ESKF
{
public:
    static constexpr int kStateDim = 15;
    static constexpr int kNoiseDim = 12;
    static constexpr int kMeasDim = 6;

    IMU_SR_ESKF()
        : position_(Eigen::Vector3d::Zero()),
          velocity_(Eigen::Vector3d::Zero()),
          orientation_(Eigen::Quaterniond::Identity()),
          gyro_bias_(Eigen::Vector3d::Zero()),
          accel_bias_(Eigen::Vector3d::Zero()),
          L_(Eigen::Matrix<double, kStateDim, kStateDim>::Identity()),
          SQ_(Eigen::Matrix<double, kNoiseDim, kNoiseDim>::Identity()),
          SR_(Eigen::Matrix<double, kMeasDim, kMeasDim>::Identity())
    {
        this->setProcessNoiseStandardDeviations(0.01, 0.1, 1e-4, 1e-4);
        this->setMeasurementNoiseStandardDeviations(0.1, 0.05);
    }

    void setState(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity,
                  const Eigen::Quaterniond &orientation, const Eigen::Vector3d &gyro_bias,
                  const Eigen::Vector3d &accel_bias)
    {
        this->position_ = position;
        this->velocity_ = velocity;
        this->orientation_ = orientation.normalized();
        this->gyro_bias_ = gyro_bias;
        this->accel_bias_ = accel_bias;
    }

    void setCovariance(const Eigen::Matrix<double, kStateDim, kStateDim> &covariance)
    {
        Eigen::LLT<Eigen::Matrix<double, kStateDim, kStateDim>> llt(covariance);
        if (llt.info() == Eigen::Success)
        {
            this->L_ = llt.matrixU();
        }
        else
        {
            this->L_.setIdentity();
        }
    }

    void setProcessNoiseStandardDeviations(double sigma_gyro, double sigma_accel,
                                           double sigma_gyro_bias, double sigma_accel_bias)
    {
        this->SQ_.setZero();
        const double s_gyro = std::max(sigma_gyro, 0.0);
        const double s_accel = std::max(sigma_accel, 0.0);
        const double s_bg = std::max(sigma_gyro_bias, 0.0);
        const double s_ba = std::max(sigma_accel_bias, 0.0);

        this->SQ_.block<3, 3>(0, 0) = s_gyro * Eigen::Matrix3d::Identity();
        this->SQ_.block<3, 3>(3, 3) = s_accel * Eigen::Matrix3d::Identity();
        this->SQ_.block<3, 3>(6, 6) = s_bg * Eigen::Matrix3d::Identity();
        this->SQ_.block<3, 3>(9, 9) = s_ba * Eigen::Matrix3d::Identity();
    }

    void setMeasurementNoiseStandardDeviations(double sigma_position, double sigma_orientation)
    {
        this->SR_.setZero();
        const double s_p = std::max(sigma_position, 0.0);
        const double s_theta = std::max(sigma_orientation, 0.0);
        this->SR_.block<3, 3>(0, 0) = s_p * Eigen::Matrix3d::Identity();
        this->SR_.block<3, 3>(3, 3) = s_theta * Eigen::Matrix3d::Identity();
    }

    Eigen::Isometry3d getPose() const
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = this->position_;
        pose.linear() = this->orientation_.toRotationMatrix();
        return pose;
    }

    const Eigen::Vector3d &getVelocity() const { return this->velocity_; }
    const Eigen::Vector3d &getGyroBias() const { return this->gyro_bias_; }
    const Eigen::Vector3d &getAccelBias() const { return this->accel_bias_; }
    Eigen::Matrix<double, kStateDim, kStateDim> getCovariance() const { return this->L_.transpose() * this->L_; }

    void predict(double dt, const Eigen::Vector3d &gyro_measurement, const Eigen::Vector3d &accel_measurement,
                 const Eigen::Vector3d &gravity)
    {
        if (dt <= 0.0)
        {
            return;
        }

        const Eigen::Vector3d velocity_prev = this->velocity_;

        const Eigen::Vector3d omega_unbiased = gyro_measurement - this->gyro_bias_;
        const Eigen::Vector3d accel_unbiased = accel_measurement - this->accel_bias_;

        const Eigen::Quaterniond delta_q = expSO3(omega_unbiased * dt);
        this->orientation_ = (this->orientation_ * delta_q).normalized();
        const Eigen::Matrix3d Rwb = this->orientation_.toRotationMatrix();
        const Eigen::Vector3d world_accel = Rwb * accel_unbiased + gravity;

        this->position_ += velocity_prev * dt + 0.5 * world_accel * dt * dt;
        this->velocity_ += world_accel * dt;

        Eigen::Matrix<double, kStateDim, kStateDim> F = Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(3, 6) = -Rwb * skew(accel_unbiased) * dt;
        F.block<3, 3>(3, 12) = -Rwb * dt;
        F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - skew(omega_unbiased) * dt;
        F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

        Eigen::Matrix<double, kStateDim, kNoiseDim> Gd = Eigen::Matrix<double, kStateDim, kNoiseDim>::Zero();
        Gd.block<3, 3>(0, 3) = -0.5 * Rwb * dt * dt;
        Gd.block<3, 3>(3, 3) = -Rwb * dt;
        Gd.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity() * dt;
        Gd.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity() * dt;
        Gd.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;

        Eigen::Matrix<double, kStateDim + kNoiseDim, kStateDim> stacked;
        stacked.topRows<kStateDim>() = this->L_ * F.transpose();
        stacked.bottomRows<kNoiseDim>() = this->SQ_ * Gd.transpose();

        const Eigen::HouseholderQR<Eigen::Matrix<double, kStateDim + kNoiseDim, kStateDim>> qr(stacked);
        this->L_ = qr.matrixQR().topRows<kStateDim>().template triangularView<Eigen::Upper>();
    }

    void update(const Eigen::Vector3d &position_measurement, const Eigen::Quaterniond &orientation_measurement)
    {
        Eigen::Quaterniond q_meas = orientation_measurement.normalized();
        if (this->orientation_.coeffs().dot(q_meas.coeffs()) < 0.0)
        {
            q_meas.coeffs() *= -1.0;
        }

        Eigen::Matrix<double, kMeasDim, 1> residual;
        residual.segment<3>(0) = position_measurement - this->position_;
        const Eigen::Quaterniond delta_q = (this->orientation_.conjugate() * q_meas).normalized();
        residual.segment<3>(3) = logSO3(delta_q);

        Eigen::Matrix<double, kMeasDim, kStateDim> H = Eigen::Matrix<double, kMeasDim, kStateDim>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

        const Eigen::Matrix<double, kStateDim, kMeasDim> LHt = this->L_ * H.transpose();

        Eigen::Matrix<double, kStateDim + kMeasDim, kMeasDim> stacked_measurement;
        stacked_measurement.topRows<kMeasDim>() = this->SR_;
        stacked_measurement.bottomRows<kStateDim>() = LHt;

        const Eigen::HouseholderQR<Eigen::Matrix<double, kStateDim + kMeasDim, kMeasDim>> qr(stacked_measurement);
        const auto T = qr.matrixQR().topRows<kMeasDim>().template triangularView<Eigen::Upper>();

        const Eigen::Matrix<double, kMeasDim, 1> y = T.transpose().solve(residual);
        const Eigen::Matrix<double, kMeasDim, 1> z = T.solve(y);

        const Eigen::Matrix<double, kStateDim, 1> delta_x = this->L_.transpose() * (LHt * z);

        this->position_ += delta_x.segment<3>(0);
        this->velocity_ += delta_x.segment<3>(3);
        const Eigen::Vector3d delta_theta = delta_x.segment<3>(6);
        this->orientation_ = (this->orientation_ * expSO3(delta_theta)).normalized();
        this->gyro_bias_ += delta_x.segment<3>(9);
        this->accel_bias_ += delta_x.segment<3>(12);

        Eigen::Matrix<double, kStateDim + kMeasDim, kStateDim> stacked_covariance =
            Eigen::Matrix<double, kStateDim + kMeasDim, kStateDim>::Zero();
        stacked_covariance.bottomRows<kStateDim>() = this->L_;

        stacked_covariance = qr.householderQ().adjoint() * stacked_covariance;
        this->L_ = stacked_covariance.bottomRows<kStateDim>().template triangularView<Eigen::Upper>();
    }

    static Eigen::Vector3d logSO3(const Eigen::Quaterniond &q)
    {
        Eigen::Quaterniond qn = q.normalized();
        if (qn.w() < 0.0)
        {
            qn.coeffs() *= -1.0;
        }
        const double w = std::clamp(qn.w(), -1.0, 1.0);
        const double angle = 2.0 * std::acos(w);
        const double sin_half = std::sqrt(std::max(1.0 - w * w, 0.0));
        if (sin_half < 1e-12)
        {
            return Eigen::Vector3d(qn.x(), qn.y(), qn.z()) * 2.0;
        }
        const Eigen::Vector3d axis(qn.x(), qn.y(), qn.z()) / sin_half;
        return axis * angle;
    }

private:
    static Eigen::Matrix3d skew(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d result;
        result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
        return result;
    }

    static Eigen::Quaterniond expSO3(const Eigen::Vector3d &theta)
    {
        const double angle = theta.norm();
        if (angle < 1e-12)
        {
            const double half = 0.5;
            return Eigen::Quaterniond(1.0, half * theta.x(), half * theta.y(), half * theta.z()).normalized();
        }
        const double half_angle = 0.5 * angle;
        const double sin_half = std::sin(half_angle);
        const Eigen::Vector3d axis = theta / angle;
        return Eigen::Quaterniond(std::cos(half_angle), axis.x() * sin_half, axis.y() * sin_half,
                                  axis.z() * sin_half)
            .normalized();
    }

    Eigen::Vector3d position_;
    Eigen::Vector3d velocity_;
    Eigen::Quaterniond orientation_;
    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d accel_bias_;

    Eigen::Matrix<double, kStateDim, kStateDim> L_;
    Eigen::Matrix<double, kNoiseDim, kNoiseDim> SQ_;
    Eigen::Matrix<double, kMeasDim, kMeasDim> SR_;
};


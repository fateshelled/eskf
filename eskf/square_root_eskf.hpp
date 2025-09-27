#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

// Square-root variant of the ESKF that keeps its own nominal state and
// covariance factor without inheriting from the classic ESKF class.
class SquareRootESKF
{
public:
    using Matrix12 = Eigen::Matrix<double, 12, 12>;
    using Matrix6 = Eigen::Matrix<double, 6, 6>;

    SquareRootESKF()
        : position_(Eigen::Vector3d::Zero()),
          velocity_body_(Eigen::Vector3d::Zero()),
          orientation_(Eigen::Quaterniond::Identity()),
          angular_velocity_body_(Eigen::Vector3d::Zero()),
          P_(Matrix12::Identity()),
          sqrt_P_(Matrix12::Identity()),
          Q_(Matrix6::Identity()),
          R_(Matrix6::Identity())
    {
        // Configure default noise densities.
        setProcessNoiseDensities(0.09, 1.2e-3);
        setMeasurementNoise(0.15, 0.026);
        factorizeCovariance();
    }

    // Set the nominal state directly.
    void setState(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity_body,
                  const Eigen::Quaterniond &orientation, const Eigen::Vector3d &angular_velocity_body)
    {
        position_ = position;
        velocity_body_ = velocity_body;
        orientation_ = orientation.normalized();
        angular_velocity_body_ = angular_velocity_body;
    }

    // Assign a full covariance matrix.
    void setCovariance(const Matrix12 &covariance)
    {
        P_ = covariance;
        factorizeCovariance();
    }

    // Provide a square-root covariance directly.
    void setSquareRootCovariance(const Matrix12 &sqrt_covariance)
    {
        sqrt_P_ = sqrt_covariance;
        P_ = symmetrize(sqrt_P_ * sqrt_P_.transpose());
        factorizeCovariance();
    }

    // Retrieve the lower-triangular square-root covariance.
    const Matrix12 &getSquareRootCovariance() const { return sqrt_P_; }

    // Retrieve the full covariance matrix.
    const Matrix12 &getCovariance() const { return P_; }

    // Configure process noise densities for velocity and angular velocity random walks.
    void setProcessNoiseDensities(double q_v, double q_omega)
    {
        Q_.setZero();
        Q_.block<3, 3>(0, 0) = q_v * Eigen::Matrix3d::Identity();
        Q_.block<3, 3>(3, 3) = q_omega * Eigen::Matrix3d::Identity();
    }

    // Configure measurement noise variances for position and attitude observations.
    void setMeasurementNoise(double sigma_p, double sigma_theta)
    {
        R_.setZero();
        R_.block<3, 3>(0, 0) = (sigma_p * sigma_p) * Eigen::Matrix3d::Identity();
        R_.block<3, 3>(3, 3) = (sigma_theta * sigma_theta) * Eigen::Matrix3d::Identity();
    }

    // Access the current pose.
    Eigen::Isometry3d getPose() const
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = position_;
        pose.linear() = orientation_.toRotationMatrix();
        return pose;
    }

    // Accessors for nominal state components.
    const Eigen::Vector3d &getVelocity() const { return velocity_body_; }
    const Eigen::Vector3d &getAngularVelocity() const { return angular_velocity_body_; }

    // Predict the state and covariance forward in time.
    void predict(double dt)
    {
        const Eigen::Matrix3d Rwb = orientation_.toRotationMatrix();

        // Propagate the nominal state.
        const Eigen::Vector3d position_increment = Rwb * velocity_body_ * dt;
        position_ += position_increment;

        const Eigen::Quaterniond delta_q = expSO3(angular_velocity_body_ * dt);
        orientation_ = (orientation_ * delta_q).normalized();

        // Linearized transition matrices.
        Matrix12 F = Matrix12::Identity();
        F.block<3, 3>(0, 3) = Rwb * dt;
        F.block<3, 3>(0, 6) = -Rwb * skew(velocity_body_) * dt;
        F.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity() * dt;

        Eigen::Matrix<double, 12, 6> G = Eigen::Matrix<double, 12, 6>::Zero();
        G.block<3, 3>(0, 0) = 0.5 * Rwb * dt * dt;
        G.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * dt;
        G.block<3, 3>(6, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
        G.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity() * dt;

        const Matrix12 predicted_covariance = F * P_ * F.transpose() + G * Q_ * G.transpose();
        P_ = symmetrize(predicted_covariance);

        factorizeCovariance();
    }

    // Correct the state with position and attitude measurements.
    void update(const Eigen::Vector3d &position_meas, const Eigen::Quaterniond &orientation_meas)
    {
        Eigen::Quaterniond q_meas = orientation_meas;
        if (orientation_.coeffs().dot(q_meas.coeffs()) < 0.0)
        {
            q_meas.coeffs() *= -1.0;
        }

        const Eigen::Vector3d residual_position = position_meas - position_;
        Eigen::Quaterniond delta_q = orientation_.conjugate() * q_meas;
        delta_q.normalize();
        const Eigen::Vector3d residual_theta = logSO3(delta_q);

        Eigen::Matrix<double, 6, 1> residual;
        residual.segment<3>(0) = residual_position;
        residual.segment<3>(3) = residual_theta;

        Eigen::Matrix<double, 6, 12> H = Eigen::Matrix<double, 6, 12>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

        const Eigen::Matrix<double, 12, 6> PHt = P_ * H.transpose();
        const Eigen::Matrix<double, 6, 6> S = H * PHt + R_;
        const Eigen::Matrix<double, 6, 6> S_inv = S.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
        const Eigen::Matrix<double, 12, 6> K = PHt * S_inv;

        const Eigen::Matrix<double, 12, 1> delta_x = K * residual;

        position_ += delta_x.segment<3>(0);
        velocity_body_ += delta_x.segment<3>(3);
        const Eigen::Vector3d delta_theta = delta_x.segment<3>(6);
        orientation_ = (orientation_ * expSO3(delta_theta)).normalized();
        angular_velocity_body_ += delta_x.segment<3>(9);

        const Matrix12 IKH = Matrix12::Identity() - K * H;
        const Matrix12 updated_covariance = IKH * P_ * IKH.transpose() + K * R_ * K.transpose();
        P_ = symmetrize(updated_covariance);

        factorizeCovariance();
    }

private:
    static Matrix12 symmetrize(const Matrix12 &matrix)
    {
        return 0.5 * (matrix + matrix.transpose());
    }

    static Eigen::Matrix3d skew(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d result;
        result << 0.0, -v.z(), v.y(),
                  v.z(), 0.0, -v.x(),
                 -v.y(), v.x(), 0.0;
        return result;
    }

    static Eigen::Quaterniond expSO3(const Eigen::Vector3d &theta)
    {
        const double angle = theta.norm();
        Eigen::Quaterniond q;
        if (angle < 1e-8)
        {
            const Eigen::Vector3d half = 0.5 * theta;
            q.w() = 1.0;
            q.vec() = half;
        }
        else
        {
            const double half_angle = 0.5 * angle;
            const double sin_half = std::sin(half_angle);
            const Eigen::Vector3d axis = theta / angle;
            q.w() = std::cos(half_angle);
            q.vec() = axis * sin_half;
        }
        return q.normalized();
    }

    static Eigen::Vector3d logSO3(const Eigen::Quaterniond &q)
    {
        Eigen::Quaterniond qn = q.normalized();
        if (qn.w() < 0.0)
        {
            qn.coeffs() *= -1.0;
        }
        const double w = std::clamp(qn.w(), -1.0, 1.0);
        const double sin_half = std::sqrt(std::max(1.0 - w * w, 0.0));
        if (sin_half < 1e-8)
        {
            return 2.0 * qn.vec();
        }
        const double angle = 2.0 * std::atan2(sin_half, w);
        const Eigen::Vector3d axis = qn.vec() / sin_half;
        return axis * angle;
    }

    Matrix12 enforcePositiveSemidefinite(const Matrix12 &matrix) const
    {
        Eigen::SelfAdjointEigenSolver<Matrix12> eig(matrix);
        Eigen::Matrix<double, 12, 1> clamped = eig.eigenvalues().cwiseMax(1e-9);
        return eig.eigenvectors() * clamped.asDiagonal() * eig.eigenvectors().transpose();
    }

    void factorizeCovariance()
    {
        Matrix12 sym = symmetrize(P_);
        Eigen::LLT<Matrix12> llt(sym);
        if (llt.info() != Eigen::Success)
        {
            sym = enforcePositiveSemidefinite(sym);
            llt.compute(sym);
        }
        if (llt.info() != Eigen::Success)
        {
            sqrt_P_ = Matrix12::Identity();
            P_ = Matrix12::Identity();
            return;
        }
        sqrt_P_ = llt.matrixL();
        P_ = symmetrize(sqrt_P_ * sqrt_P_.transpose());
    }

    Eigen::Vector3d position_;
    Eigen::Vector3d velocity_body_;
    Eigen::Quaterniond orientation_;
    Eigen::Vector3d angular_velocity_body_;
    Matrix12 P_;
    Matrix12 sqrt_P_;
    Matrix6 Q_;
    Matrix6 R_;
};

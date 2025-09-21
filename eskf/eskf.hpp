#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Error-State Kalman Filter (ESKF) の実装。
// 12次元の状態ベクトル（位置、速度、向き、角速度）を推定します。
class ESKF
{
public:
    ESKF()
        : position_(Eigen::Vector3d::Zero()),
          velocity_body_(Eigen::Vector3d::Zero()),
          orientation_(Eigen::Quaterniond::Identity()),
          angular_velocity_body_(Eigen::Vector3d::Zero()),
          P_(Eigen::Matrix<double, 12, 12>::Identity()),
          Q_(Eigen::Matrix<double, 6, 6>::Identity()),
          R_(Eigen::Matrix<double, 6, 6>::Identity())
    {
        // デフォルトのノイズパラメータを設定
        setProcessNoiseDensities(0.09, 1.2e-3);
        setMeasurementNoise(0.15, 0.026);
    }

    // フィルタの初期状態を設定します。
    // @param position 初期位置
    // @param velocity_body 機体座標系での初期速度
    // @param orientation 初期姿勢
    // @param angular_velocity_body 機体座標系での初期角速度
    void setState(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity_body,
                  const Eigen::Quaterniond &orientation, const Eigen::Vector3d &angular_velocity_body)
    {
        this->position_ = position;
        this->velocity_body_ = velocity_body;
        this->orientation_ = orientation.normalized();
        this->angular_velocity_body_ = angular_velocity_body;
    }

    // 誤差状態の共分散行列を設定します。
    void setCovariance(const Eigen::Matrix<double, 12, 12> &covariance) { this->P_ = covariance; }

    // プロセスノイズの密度を設定します。
    void setProcessNoiseDensities(double q_v, double q_omega)
    {
        this->Q_.setZero();
        this->Q_.block<3, 3>(0, 0) = q_v * Eigen::Matrix3d::Identity();
        this->Q_.block<3, 3>(3, 3) = q_omega * Eigen::Matrix3d::Identity();
    }

    // 観測ノイズの標準偏差を設定します。
    void setMeasurementNoise(double sigma_p, double sigma_theta)
    {
        this->R_.setZero();
        this->R_.block<3, 3>(0, 0) = (sigma_p * sigma_p) * Eigen::Matrix3d::Identity();
        this->R_.block<3, 3>(3, 3) = (sigma_theta * sigma_theta) * Eigen::Matrix3d::Identity();
    }

    // 現在の姿勢（位置と向き）をEigen::Isometry3dとして取得します。
    Eigen::Isometry3d getPose() const
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = this->position_;
        pose.linear() = this->orientation_.toRotationMatrix();
        return pose;
    }

    // 各状態量と共分散行列を取得するゲッター
    const Eigen::Vector3d &getVelocity() const { return this->velocity_body_; }
    const Eigen::Vector3d &getAngularVelocity() const { return this->angular_velocity_body_; }
    const Eigen::Matrix<double, 12, 12> &getCovariance() const { return this->P_; }

    // 予測ステップ：次のタイムステップの状態を予測します。
    // @param dt 経過時間
    void predict(double dt)
    {
        const Eigen::Matrix3d R = this->orientation_.toRotationMatrix();

        // 1. 公称状態の予測
        {
            // 位置の更新
            const Eigen::Vector3d position_increment = R * this->velocity_body_ * dt;
            this->position_ += position_increment;

            // 姿勢の更新
            const Eigen::Quaterniond delta_q = expSO3(this->angular_velocity_body_ * dt);
            this->orientation_ = (this->orientation_ * delta_q).normalized();
        }

        // update noise covariance
        {
            Eigen::Matrix<double, 12, 12> F = Eigen::Matrix<double, 12, 12>::Identity();
            // 誤差状態遷移行列 F の構築
            F.block<3, 3>(0, 3) = R * dt;
            F.block<3, 3>(0, 6) = -R * skew(this->velocity_body_) * dt;
            F.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity() * dt;

            Eigen::Matrix<double, 12, 6> G = Eigen::Matrix<double, 12, 6>::Zero();
            // ノイズ駆動行列 G の構築
            G.block<3, 3>(0, 0) = 0.5 * R * dt * dt;
            G.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * dt;
            G.block<3, 3>(6, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
            G.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity() * dt;

            const Eigen::Matrix<double, 12, 12> FP = F * this->P_;
            this->P_ = FP * F.transpose() + G * this->Q_ * G.transpose();
            // 対称性を維持するための処理
            this->P_ = 0.5 * (P_ + this->P_.transpose());
        }
    }

    // 更新ステップ：観測値を用いて状態を補正します。
    // @param position_meas 観測された位置
    // @param orientation_meas 観測された姿勢
    void update(const Eigen::Vector3d &position_meas, const Eigen::Quaterniond &orientation_meas)
    {
        Eigen::Quaterniond q_meas = orientation_meas;
        // クォータニオンの符号の曖昧性を解決
        if (this->orientation_.coeffs().dot(q_meas.coeffs()) < 0.0)
        {
            q_meas.coeffs() *= -1.0;
        }

        // 1. 観測残差の計算
        const Eigen::Vector3d residual_position = position_meas - this->position_;
        Eigen::Quaterniond delta_q = this->orientation_.conjugate() * q_meas;
        delta_q.normalize();
        const Eigen::Vector3d residual_theta = logSO3(delta_q);

        // 6次元の残差ベクトル
        Eigen::Matrix<double, 6, 1> residual;
        residual.segment<3>(0) = residual_position;
        residual.segment<3>(3) = residual_theta;

        // 2. 観測行列 H の構築
        Eigen::Matrix<double, 6, 12> H = Eigen::Matrix<double, 6, 12>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

        const Eigen::Matrix<double, 12, 6> PHt = this->P_ * H.transpose();
        const Eigen::Matrix<double, 6, 6> S = H * PHt + this->R_;
        const Eigen::Matrix<double, 6, 6> S_inv = S.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
        const Eigen::Matrix<double, 12, 6> K = PHt * S_inv;

        // 4. 誤差状態の補正量を計算
        Eigen::Matrix<double, 12, 1> delta_x = K * residual;

        // 5. 公称状態の補正 (Inject error state into nominal state)
        this->position_ += delta_x.segment<3>(0);
        this->velocity_body_ += delta_x.segment<3>(3);
        const Eigen::Vector3d delta_theta = delta_x.segment<3>(6);
        this->orientation_ = (this->orientation_ * expSO3(delta_theta)).normalized();
        this->angular_velocity_body_ += delta_x.segment<3>(9);

        // 6. 共分散行列の更新 (Joseph form)
        const Eigen::Matrix<double, 12, 12> IKH = Eigen::Matrix<double, 12, 12>::Identity() - K * H;
        this->P_ = IKH * this->P_ * IKH.transpose() + K * this->R_ * K.transpose();
        // 対称性を維持するための処理
        this->P_ = 0.5 * (this->P_ + this->P_.transpose());
    }

private:
    // 3次元ベクトルから歪対称行列（skew-symmetric matrix）を生成します。
    static Eigen::Matrix3d skew(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d result;
        result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
        return result;
    }

    // SO(3)の指数写像：回転ベクトルをクォータニオンに変換します。
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

    // SO(3)の対数写像：クォータニオンを回転ベクトルに変換します。
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

    // --- 状態変数 ---
    Eigen::Vector3d position_;              // 位置 (ワールド座標系)
    Eigen::Vector3d velocity_body_;         // 速度 (機体座標系)
    Eigen::Quaterniond orientation_;        // 姿勢 (ワールド座標系から機体座標系への回転)
    Eigen::Vector3d angular_velocity_body_; // 角速度 (機体座標系)
    Eigen::Matrix<double, 12, 12> P_;       // 誤差状態の共分散行列
    Eigen::Matrix<double, 6, 6> Q_;         // プロセスノイズの共分散行列
    Eigen::Matrix<double, 6, 6> R_;         // 観測ノイズの共分散行列
};

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

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
          L_(Eigen::Matrix<double, 12, 12>::Identity()),
          SQ_(Eigen::Matrix<double, 6, 6>::Identity()),
          SR_(Eigen::Matrix<double, 6, 6>::Identity())
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
    void setCovariance(const Eigen::Matrix<double, 12, 12> &covariance)
    {
        Eigen::LLT<Eigen::Matrix<double, 12, 12>> llt(covariance);
        if (llt.info() == Eigen::Success)
        {
            this->L_ = llt.matrixU(); // upper triangle
        }
        else
        {
            this->L_.setIdentity();
        }
    }

    // プロセスノイズの密度を設定します。
    void setProcessNoiseDensities(double q_v, double q_omega)
    {
        Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = q_v * Eigen::Matrix3d::Identity();
        Q.block<3, 3>(3, 3) = q_omega * Eigen::Matrix3d::Identity();

        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(Q);
        if (llt.info() == Eigen::Success)
        {
            this->SQ_ = llt.matrixU(); // upper triangle
        }
        else
        {
            this->SQ_.setZero();
        }
    }

    // 観測ノイズの標準偏差を設定します。
    void setMeasurementNoise(double sigma_p, double sigma_theta)
    {
        Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
        R.block<3, 3>(0, 0) = (sigma_p * sigma_p) * Eigen::Matrix3d::Identity();
        R.block<3, 3>(3, 3) = (sigma_theta * sigma_theta) * Eigen::Matrix3d::Identity();

        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(R);
        if (llt.info() == Eigen::Success)
        {
            this->SR_ = llt.matrixU(); // upper triangle
        }
        else
        {
            this->SR_.setZero();
        }
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
    Eigen::Matrix<double, 12, 12> getCovariance() const { return this->L_.transpose() * this->L_; }

    // 予測ステップ：次のタイムステップの状態を予測します。
    // @param dt 経過時間
    void predict(double dt)
    {
        if (dt <= 0.0) return;

        const Eigen::Matrix3d R = this->orientation_.toRotationMatrix();

        // 1. 公称状態の予測
        {
            // 定速・定角速度モデル
            // 速度と角速度はプロセスノイズによるランダムウォーク

            // 位置の更新
            // R * v: world座標系速度
            const Eigen::Vector3d position_increment = R * this->velocity_body_ * dt;
            this->position_ += position_increment;

            // 姿勢の更新
            const Eigen::Quaterniond delta_q = expSO3(this->angular_velocity_body_ * dt);
            this->orientation_ = (this->orientation_ * delta_q).normalized();
        }

        // 2. ノイズ共分散行列の更新
        {
            // 誤差状態遷移行列 F の構築
            Eigen::Matrix<double, 12, 12> F = Eigen::Matrix<double, 12, 12>::Identity();
            {
                //                    position    velocity       quaternion       angular_velocity
                //         position |     I    |   R * dt  |  -R * skew(v) * dt |                  |
                //         velocity |          |     I     |                    |                  |
                //       quaternion |          |           |         I          |      I * dt      |
                // angular_velocity |          |           |                    |        I         |
                F.block<3, 3>(0, 3) = R * dt;
                F.block<3, 3>(0, 6) = -R * skew(this->velocity_body_) * dt;
                F.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity() * dt;
            }

            // ノイズ駆動行列 G の構築
            Eigen::Matrix<double, 12, 6> G = Eigen::Matrix<double, 12, 6>::Zero();
            {
                //                        position         velocity    quaternion   angular_velocity
                //         position |   0.5 * R * dt^2  |            |            |                  |
                //         velocity |       I * dt      |    Zero    |            |                  |
                //       quaternion |                   |            |    Zero    |                  |
                // angular_velocity |                   |   I * dt   |            |        Zero      |
                G.block<3, 3>(0, 0) = 0.5 * R * dt * dt;
                G.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * dt;
                G.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity() * dt;
            }

            // 誤差共分散行列の平方根因子 L をQR分解を使って求める
            // P: 誤差共分散行列 ( = L.T * L )
            // Q: プロセスノイズ共分散行列 ( = SQ.T * SQ )
            // QR分解結果の R 行列が予測誤差共分散の平方根因子

            // stacked = | L * F.T  |
            //           | SQ * G.T |
            // P_pred = F * P * F.T + G * Q * G.T
            //        = F * (L.T * L) * F.T + G * (SQ.T * SQ) * G.T
            //        = (F * L.T) * (L * F.T) + (G * SQ.T) * (SQ * G.T)
            //        = | F * L.T;  G * SQ.T | * | L * F.T  |
            //                                   | SQ * G.T |
            //        = stacked.T * stacked

            // P_pred = (Q * R).T * (Q * R)
            //        = R.T * Q.T * Q * R
            //        = R.T * R
            // (Qは直交行列なので、Q.T * Q = I)

            Eigen::Matrix<double, 18, 12> stacked;
            stacked.block<12, 12>(0, 0) = this->L_ * F.transpose();
            stacked.block<6, 12>(12, 0) = this->SQ_ * G.transpose();

            const Eigen::HouseholderQR<Eigen::Matrix<double, 18, 12>> qr(stacked);
            this->L_ = qr.matrixQR().topRows<12>().template triangularView<Eigen::Upper>();
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
        // 6次元の残差ベクトル
        Eigen::Matrix<double, 6, 1> residual;
        {
            const Eigen::Vector3d residual_position = position_meas - this->position_;
            const Eigen::Quaterniond delta_q = (this->orientation_.conjugate() * q_meas).normalized();
            const Eigen::Vector3d residual_theta = logSO3(delta_q);

            residual.segment<3>(0) = residual_position;
            residual.segment<3>(3) = residual_theta;
        }

        // 2. 観測行列 H の構築
        Eigen::Matrix<double, 6, 12> H = Eigen::Matrix<double, 6, 12>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

        // 3. カルマンゲイン相当を計算
        const Eigen::Matrix<double, 12, 6> LHt = this->L_ * H.transpose();

        Eigen::Matrix<double, 18, 6> stacked_measurement;
        {
            // R: 観測ノイズ共分散 ( = SR.T * SR)
            // stacked_measurement = |    SR   |
            //                       | L * H.T |
            // S  = stacked_measurement.T * stacked_measurement
            //    = | SR.T ; H * L.T| * |    SR   |
            //                          | L * H.T |
            //    = SR.T * SR + H * L.T * L * H.T
            //    = H * P(prev) * H.T + R

            stacked_measurement.block<6, 6>(0, 0) = this->SR_;
            stacked_measurement.block<12, 6>(6, 0) = LHt;
        }

        // stacked_measurementをQR分解することで、行列 S を求めずに、逆行列 S.inv を残差 residual にかけたベクトルを直接求める
        const Eigen::HouseholderQR<Eigen::Matrix<double, 18, 6>> qr(stacked_measurement);
        // S の 平方根因子
        const auto T = qr.matrixQR().topRows<6>().template triangularView<Eigen::Upper>();
        // solve: T.T * y = residual
        const Eigen::Matrix<double, 6, 1> y = T.transpose().solve(residual);
        // solve: T * z = y
        const Eigen::Matrix<double, 6, 1> z = T.solve(y);

        // T.T * y = residualにy = T * zを代入
        // T.T * (T * z) = residual
        // (T.T * T) * z = residual
        // S * z = residual
        // z = S.inv * residual

        // 4. 誤差状態の補正量を計算
        // K: カルマンゲイン
        // delta_x = K * z
        //         = P * H.T * S.inv * z
        const Eigen::Matrix<double, 12, 1> delta_x = this->L_.transpose() * (LHt * z);

        // 5. 公称状態の補正
        {
            this->position_ += delta_x.segment<3>(0);
            this->velocity_body_ += delta_x.segment<3>(3);
            const Eigen::Vector3d delta_theta = delta_x.segment<3>(6);
            this->orientation_ = (this->orientation_ * expSO3(delta_theta)).normalized();
            this->angular_velocity_body_ += delta_x.segment<3>(9);
        }

        // 6. 共分散行列の更新
        // stacked_covariance = | 0 |
        //                      | L |
        Eigen::Matrix<double, 18, 12> stacked_covariance = Eigen::Matrix<double, 18, 12>::Zero();
        stacked_covariance.bottomRows<12>() = this->L_;

        // 左から Q.T をかける
        stacked_covariance = qr.householderQ().adjoint() * stacked_covariance;
        // 下ブロックの行列が更新後の共分散平方根因子
        this->L_ = stacked_covariance.bottomRows<12>().template triangularView<Eigen::Upper>();

        // この実装では
        // stacked = |    SR   ; 0 | に対して、左右に分割してそれぞれ処理を行っている。
        //           | L * H.T ; L |
        // 上記のstackedをQR分解し、得られたR行列が R = | R11; R12 | となり、更新後の共分散平方根因子はR22となる。
        //                                              |   0; R22 |
        // 6の共分散行列の更新では、stacked = Q * R となるので、Qは直交行列であることから左からQ.Tをかけることで、R を求めている
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
    Eigen::Quaterniond orientation_;        // 姿勢 (機体座標系からワールド座標系への回転)
    Eigen::Vector3d angular_velocity_body_; // 角速度 (機体座標系)
    Eigen::Matrix<double, 12, 12> L_;       // 誤差状態共分散の平方根（上三角）
    Eigen::Matrix<double, 6, 6> SQ_;        // プロセスノイズ共分散の平方根（上三角）
    Eigen::Matrix<double, 6, 6> SR_;        // 観測ノイズ共分散の平方根（上三角）
};

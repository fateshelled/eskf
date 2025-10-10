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
        this->SQ_.setZero();
        const double sigma_v = std::sqrt(std::max(q_v, 0.0));
        const double sigma_omega = std::sqrt(std::max(q_omega, 0.0));
        this->SQ_.block<3, 3>(0, 0) = sigma_v * Eigen::Matrix3d::Identity();
        this->SQ_.block<3, 3>(3, 3) = sigma_omega * Eigen::Matrix3d::Identity();
    }

    // 観測ノイズの標準偏差を設定します。
    void setMeasurementNoise(double sigma_p, double sigma_theta)
    {
        this->SR_.setZero();
        this->SR_.block<3, 3>(0, 0) = std::abs(sigma_p) * Eigen::Matrix3d::Identity();
        this->SR_.block<3, 3>(3, 3) = std::abs(sigma_theta) * Eigen::Matrix3d::Identity();
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
        if (dt <= 0.0)
        {
            return;
        }

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
            //  次の誤差状態 = F * 現在の誤差状態 + G * プロセスノイズ

            //  誤差状態方程式は次のように表される：
            //      dδx/dt = F * δx + G * w
            //  w はプロセスノイズベクトルであり、以下のように定義する
            //      w = [ w_v, w_ω ]^T
            //      w_v : 速度のランダムウォークを駆動するノイズ（擬似的な加速度ノイズ）
            //      w_ω : 角速度のランダムウォークを駆動するノイズ（擬似的な角加速度ノイズ）

            //  δxを t -> t + Δt の間で一次近似の積分をすると
            //      δx(t + Δt) ≒ δx(t) + dδx(t)/dt * Δt
            //  この式をESKFの文脈に書き換えると、
            //  時刻 t + Δt の誤差状態は予測後の状態、δxの時間微分は誤差状態方程式となる
            //      δx(pred) ≒ δx(current) + (F * δx + G * w) * Δt
            //
            //  モデル上の仮定：
            //      ・速度 v は一定だが、ランダムウォークノイズ w_v によりゆらぐ
            //      ・角速度 ω も一定だが、ランダムウォークノイズ w_ω によりゆらぐ
            //

            // 誤差状態遷移行列 F の構築
            // （真値の運動方程式と公称状態の差を一次近似したヤコビアン）
            Eigen::Matrix<double, 12, 12> F = Eigen::Matrix<double, 12, 12>::Identity();
            {
                // 1行目: 誤差モデル p = p^ + δp
                //        運動方程式 真値 p(pred) = p(current) + R * v * dt + 1/2 * R * w_v * dt^2
                //                   公称 p^(pred) = p^(current) + R^ * v^ * dt
                //        真値と公称状態の運動方程式の差分をとると、
                //        p(pred) - p^(pred) = p(current) - p^(current) + R * v * dt - R^ * v^ * dt + 1/2 * R * w_v * dt^2
                //        ここで ^ は公称状態であるということ、真値と公称の差分をδ、歪対称行列を []x で表す
                //        真値のRが未知だが、R ≒ R^ * (I + [δθ]x) と近似する
                //        δp(pred) = δp(current) + R^ * (I + [δθ]x) * v * dt - R^ * v^ * dt + 1/2 * R^ * (I + [δθ]x) * w_v * dt^2
                //                 = δp(current) + R^ * (v - v^) * dt + R^ * [δθ]x * v * dt + 1/2 * R^ * w_v * dt^2 + 1/2 * R^ * [δθ]x * w_v * dt^2
                //                 = δp(current) + R^ * δv * dt + R^ * [δθ]x * v * dt + 1/2 * R^ * w_v * dt^2 + 1/2 * R^ * [δθ]x * w_v * dt^2
                //        ここで、[δθ]x * v は δθとvのクロス積なので、[δθ]x * v = -[v]x * δθ
                //        また、[δθ]x * w_v は誤差とノイズの積なので [δθ]x * w_v ≒ 0 とすると、
                //        δp(pred) = δp(current) + R^ * δv * dt - R^ * [v]x * δθ * dt + 1/2 * R^ * w_v * dt^2
                //        δvが十分に小さいとすると、v = v^ + δv ≒ v^ なので
                //        δp(pred) = δp(current) + R^ * δv * dt - R^ * [v^]x * δθ * dt + 1/2 * R^ * w_v * dt^2
                //        したがって、
                //        ∂δp(pred) / ∂δv = R^ * dt, ∂δp(pred) / ∂δθ = - R^ * [v^]x * dt

                // 2行目: 誤差モデル v(pred) = v(current)^ + δv(current)
                //        運動方程式 真値 v(pred) = v(current) + w_v * dt
                //                   公称 v^(pred) = v^(current)
                //        1行目と同様に
                //        v(prec) - v^(pred) = v(current) - v^(current) + w_v * dt
                //        δv(pred) = δv(current) + w_v * dt
                //        したがって、
                //        ∂δv(pred) / ∂δv(current) = I

                // 3行目: 誤差モデル q(pred) = q(pred)^ ⊗ exp(1/2 * δθ(pred))                                公称姿勢に姿勢誤差を加えることで、真値姿勢が得られる
                //                   q(current) = q(current)^ ⊗ exp(1/2 * δθ(current))
                //        運動方程式 q(pred) = q(current) ⊗ exp(1/2 * (ω(current) * dt + 1/2 * w_ω * dt^2))   現在姿勢に角速度の時間積分値を加えると次時刻の姿勢が得られる
                //                   q(pred)^ = q(current)^ ⊗ exp(1/2 * ω(current)^ * dt)
                //        誤差モデルに対して、左から (q(pred)^).inv をかけると
                //        exp(1/2 δθ(pred)) = (q(pred)^).inv ⊗ q(pred)
                //                          = (q(current)^ ⊗ exp(1/2 * ω(current)^ * dt)).inv ⊗ (q(current) ⊗ exp(1/2 * (ω(current) * dt + 1/2 * w_ω * dt^2)))
                //        ω = ω^ + δωなので
                //        exp(1/2 δθ(pred)) = (q(current)^ ⊗ exp(1/2 * ω(current)^ * dt)).inv    ⊗ (q(current) ⊗ exp(1/2 * ((ω(current)^ + δω(current)) * dt  + 1/2 * w_ω * dt^2)))
                //        exp(1/2 δθ(pred)) = (exp(-1/2 * ω(current)^ * dt) ⊗ (q(current)^).inv) ⊗ (q(current) ⊗ exp(1/2 * (ω(current)^ * dt + δω(current)* dt  + 1/2 * w_ω * dt^2)))
                //        誤差モデルq(current)~を用いて q(current) と q(current)^ を置き換えて
                //        exp(1/2 δθ(pred)) = (exp(-1/2 * ω(current)^ * dt) ⊗ (q(current)^).inv) ⊗ ((q(current)^ ⊗ exp(1/2 δθ(current))) ⊗ exp(1/2 * (ω(current)^ * dt + δω(current)* dt  + 1/2 * w_ω * dt^2)))
                //                          =  exp(-1/2 * ω(current)^ * dt) ⊗ exp(1/2 * δθ(current)) ⊗ exp(1/2 * (ω(current)^ * dt + δω(current)* dt  + 1/2 * w_ω * dt^2)))
                //        ここでexp(A) ⊗ exp(B) ≒ exp(A + B)という一次近似（BCH（Baker–Campbell–Hausdorff）展開）を用いて右辺をまとめると、
                //        exp(1/2 δθ(pred)) = exp(1/2 * (δθ(current) + δω(current) * dt + 1/2 w_ω * dt^2))
                //        expの中だけ抜き出して、
                //        δθ(pred) = δθ(current) + δω(current) * dt + 1/2 * w_ω * dt^2
                //        したがって
                //        ∂δθ(pred) / ∂δθ(current) = I, ∂δθ(pred) / ∂δω(current) = I * dt

                // 4行目: 誤差モデル ω(pred) = ω(current)^ + δω(current)
                //        運動方程式 真値 ω(pred) = ω(current) + w_ω * dt
                //                   公称 ω^(pred) = ω^(current)
                //        2行目と同様に
                //        ω(prec) - ω^(pred) = ω(current) - ω^(current) + w_ω * dt
                //        δω(pred) = δω(current) + w_ω * dt
                //        したがって、
                //        ∂δω(pred) / ∂δω(current) = I

                //  pred \ current      δp         δv              δθ                 δω
                //             δp  |     I    |   R^ * dt  |  -R^ * [v^]x * dt |                  |
                //             δv  |          |     I      |                   |                  |
                //             δθ  |          |            |         I         |      I * dt      |
                //             δω |          |            |                   |        I         |

                F.block<3, 3>(0, 3) = R * dt;
                F.block<3, 3>(0, 6) = -R * skew(this->velocity_body_) * dt;
                F.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity() * dt;
            }

            // ノイズ駆動行列 G の構築
            Eigen::Matrix<double, 12, 6> G = Eigen::Matrix<double, 12, 6>::Zero();
            {
                //  1行目: 位置に関してF行列で導入した下記の式についてw_vとw_ωで微分すると
                //         δp(pred) = δp(current) + R^ * δv(current) * dt - R^ * [v^]x * δθ(current) * dt + 1/2 * R^ * w_v * dt^2
                //         ∂δp(pred) / ∂w_v = 1/2 * R^ * dt^2
                //         ∂δp(pred) / ∂w_ω = 0
                //
                //  2行目: δv(pred) = δv(current) + w_v * dt
                //         ∂δv(pred) / ∂w_v = I * dt
                //         ∂δv(pred) / ∂w_ω = 0
                //
                //  3行目: δθ(pred) = δθ(current) + δω(current) * dt + 1/2 * w_ω * dt^2
                //         ∂θ(pred) / ∂w_v = 0
                //         ∂θ(pred) / ∂w_ω = 1/2 * I * dt^2
                //
                //  4行目: δω(pred) = δω(current) + w_ω * dt
                //         ∂δω(pred) / ∂w_v = 0
                //         ∂δω(pred) / ∂w_ω = I * dt
                //
                //                w_v                 w_ω
                //    δp |   1/2 * R * dt^2  |                    |
                //    δv |       I * dt      |                    |
                //  　δθ |                   |   1/2 * I * dt^2   |
                //    δω |                   |      I * dt        |

                G.block<3, 3>(0, 0) = 0.5 * R * dt * dt;
                G.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * dt;
                G.block<3, 3>(6, 3) = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
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

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>

/**
 * @brief Error State Kalman Filter
 *
 * 状態: 位置、速度、姿勢（等速・等角速度運動モデル）
 * 観測: 6DoF姿勢
 *
 */
class ErrorStateKF
{
private:
    // ===================== 型定義 =====================
    static constexpr int ERROR_STATE_DIM = 12; // 誤差状態次元数
    static constexpr int OBSERVATION_DIM = 6;  // 観測次元数

    using ErrorStateVector = Eigen::Vector<double, ERROR_STATE_DIM>;
    using ObservationVector = Eigen::Vector<double, OBSERVATION_DIM>;
    using ErrorCovariance = Eigen::Matrix<double, ERROR_STATE_DIM, ERROR_STATE_DIM>;
    using StateTransition = Eigen::Matrix<double, ERROR_STATE_DIM, ERROR_STATE_DIM>;
    using ProcessNoise = Eigen::Matrix<double, ERROR_STATE_DIM, ERROR_STATE_DIM>;
    using ObservationJacobian = Eigen::Matrix<double, OBSERVATION_DIM, ERROR_STATE_DIM>;
    using ObservationCovariance = Eigen::Matrix<double, OBSERVATION_DIM, OBSERVATION_DIM>;
    using InnovationCovariance = Eigen::Matrix<double, OBSERVATION_DIM, OBSERVATION_DIM>;
    using KalmanGain = Eigen::Matrix<double, ERROR_STATE_DIM, OBSERVATION_DIM>;

public:
    // ===================== 状態定義 =====================

    /** 名目状態 (16次元) */
    struct NominalState
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();             // p_n
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();             // v_n
        Eigen::Quaterniond quaternion = Eigen::Quaterniond::Identity(); // q_n
        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();     // ω_n

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    /** 誤差状態 (12次元) */
    struct ErrorState
    {
        Eigen::Vector3d delta_position = Eigen::Vector3d::Zero();         // δp
        Eigen::Vector3d delta_velocity = Eigen::Vector3d::Zero();         // δv
        Eigen::Vector3d delta_orientation = Eigen::Vector3d::Zero();      // δθ (軸角)
        Eigen::Vector3d delta_angular_velocity = Eigen::Vector3d::Zero(); // δω

        /** ベクトル形式に変換 */
        ErrorStateVector toVector() const
        {
            ErrorStateVector vec;
            vec.template segment<3>(0) = delta_position;
            vec.template segment<3>(3) = delta_velocity;
            vec.template segment<3>(6) = delta_orientation;
            vec.template segment<3>(9) = delta_angular_velocity;
            return vec;
        }

        /** ベクトルから設定 */
        void fromVector(const ErrorStateVector &vec)
        {
            delta_position = vec.template segment<3>(0);
            delta_velocity = vec.template segment<3>(3);
            delta_orientation = vec.template segment<3>(6);
            delta_angular_velocity = vec.template segment<3>(9);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    // ===================== コンストラクタ =====================

    ErrorStateKF()
    {
        this->initializeParameters();
        this->initializeWorkingMatrices();
        this->reset();
    }

    // ===================== メイン処理 =====================

    /**
     * @brief 観測更新（タイムスタンプから自動でdt計算・予測実行）
     * @param odom 6DoF姿勢
     * @param timestamp 現在のタイムスタンプ [sec]
     */
    void update(const Eigen::Isometry3d &odom,
                const double timestamp)
    {

        // タイムスタンプからdt計算・予測実行
        if (this->last_timestamp_ > 0.0)
        {
            const double dt = timestamp - this->last_timestamp_;

            // 負のdtチェック
            if (dt < 0.0)
            {
                std::cerr << "Error: Negative dt = " << dt << " sec. Timestamps must be monotonic. Update skipped." << std::endl;
                return;
            }

            // 異常なdt値のチェックと予測の実行
            if (dt > EPSILON_TIME && dt < DT_WARNING_THRESHOLD)
            {
                this->predict(dt);
            }
            else if (dt < EPSILON_TIME)
            {
                // dtがゼロに近い場合は予測をスキップ（同じタイムスタンプでの再更新など）
                // 観測更新は行われる
            }
            else
            {
                std::cerr << "Warning: Abnormal dt = " << dt << " sec (must be > "
                          << EPSILON_TIME << " and < " << DT_WARNING_THRESHOLD << "). Skipping prediction." << std::endl;
            }
        }
        this->last_timestamp_ = timestamp;

        try
        {
            // 固定観測共分散行列を使用
            const ObservationCovariance measurement_cov = this->getDefaultMeasurementCovariance();

            // 6DoF観測ベクトル作成
            ObservationVector z_meas;
            z_meas.template head<3>() = odom.translation();
            z_meas.template tail<3>() = this->quaternionToAxisAngle(Eigen::Quaterniond(odom.rotation()));

            // 観測予測値
            ObservationVector h_pred;
            h_pred.template head<3>() = this->nominal_state_.position;
            // 姿勢の観測予測値は、誤差状態モデルにおいて名目状態からの偏差が観測されるためゼロとなる
            h_pred.template tail<3>() = Eigen::Vector3d::Zero();

            // 残差計算（姿勢は特別処理）
            ObservationVector innovation;
            innovation.template head<3>() = z_meas.template head<3>() - h_pred.template head<3>();

            // 姿勢残差: manifold上での計算 (z_quat = q_n * delta_q_z_inv, so innovation_quat = q_n.inverse * z_quat = delta_q_z_inv)
            const Eigen::Quaterniond z_quat = this->axisAngleToQuaternion(z_meas.template tail<3>());
            const Eigen::Quaterniond innovation_quat = this->nominal_state_.quaternion.inverse() * z_quat;
            innovation.template tail<3>() = this->quaternionToAxisAngle(innovation_quat);

            // 観測ヤコビアン
            this->computeObservationJacobian(this->H_temp_);

            // カルマン更新
            this->performKalmanUpdate(this->H_temp_, innovation, measurement_cov);

            // 誤差注入とリセット
            this->injectErrorAndReset();

            // デバッグログ（必要に応じて）
            if (this->enable_logging_)
            {
                this->logState("After Update");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in LiDAR update: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * @brief 状態リセット
     */
    void reset()
    {
        this->nominal_state_ = NominalState{};
        this->error_state_ = ErrorState{};
        this->error_covariance_ = ErrorCovariance::Identity() * INITIAL_COV_SMALL_VALUE;
        this->last_timestamp_ = -1.0;  // 未初期化を示す
        this->enable_logging_ = false; // デフォルトでログ無効
    }

    // ===================== アクセサ =====================

    /** 現在の姿勢取得 */
    Eigen::Isometry3d getCurrentPose() const
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = this->nominal_state_.position;
        pose.linear() = this->nominal_state_.quaternion.toRotationMatrix();
        return pose;
    }

    /** 名目状態取得 */
    const NominalState &getNominalState() const
    {
        return this->nominal_state_;
    }

    /** 誤差共分散取得 */
    const ErrorCovariance &getErrorCovariance() const
    {
        return this->error_covariance_;
    }

    /** 位置の不確実性取得 (標準偏差) */
    Eigen::Vector3d getPositionUncertainty() const
    {
        return this->error_covariance_.template block<3, 3>(0, 0).diagonal().cwiseSqrt();
    }

    /** 姿勢の不確実性取得 (標準偏差) [rad] */
    Eigen::Vector3d getOrientationUncertainty() const
    {
        return this->error_covariance_.template block<3, 3>(6, 6).diagonal().cwiseSqrt();
    }

    /** 共分散行列の条件数取得（数値安定性チェック用） */
    double getCovarianceConditionNumber() const
    {
        const Eigen::JacobiSVD<ErrorCovariance> svd(this->error_covariance_);
        const auto singular_values = svd.singularValues();
        if (singular_values(singular_values.size() - 1) < COVARIANCE_SINGULARITY_THRESHOLD)
        {
            return std::numeric_limits<double>::infinity();
        }
        return singular_values(0) / singular_values(singular_values.size() - 1);
    }

private:
    // ===================== 内部状態 =====================

    NominalState nominal_state_;
    ErrorState error_state_;
    ErrorCovariance error_covariance_;
    double last_timestamp_; // 前回のタイムスタンプ [sec]

    // 作業用行列（メモリ効率改善）
    mutable StateTransition F_temp_;
    mutable ProcessNoise G_temp_;
    mutable ObservationJacobian H_temp_;
    mutable ProcessNoise Q_temp_;

    // デバッグ用
    bool enable_logging_;

    // パラメータ
    struct Parameters
    {
        // プロセスノイズ（各ノイズの標準偏差）
        double position_noise = 0.01;     // 速度ランダムウォークノイズ (velocity random walk std dev) [m/s]
        double velocity_noise = 0.1;      // 速度直接ノイズ (velocity direct noise std dev) [m/s]
        double orientation_noise = 0.005; // 角速度ランダムウォークノイズ (angular velocity random walk std dev) [rad/s]
        double angular_vel_noise = 0.01;  // 角速度直接ノイズ (angular velocity direct noise std dev) [rad/s]

        // 観測ノイズ（固定値）
        double pos_noise = 0.02; // 位置観測ノイズ [m]
        double ori_noise = 0.01; // 姿勢観測ノイズ [rad]
    };
    Parameters params_;

    // ===================== 定数 =====================
    static constexpr double EPSILON_TIME = 1e-9;        // 時間の最小差分
    static constexpr double EPSILON_ANGLE = 1e-8;       // 角度の最小値
    static constexpr double DT_WARNING_THRESHOLD = 1.0; // dtの警告閾値 [sec]

    // 初期共分散設定
    static constexpr double INITIAL_COV_SMALL_VALUE = 1e-6; // デフォルト初期共分散の乗数
    static constexpr double INITIAL_POS_UNCERT_STD = 0.1;   // 初期位置不確実性の標準偏差 [m]
    static constexpr double INITIAL_ORI_UNCERT_STD = 0.1;   // 初期姿勢不確実性の標準偏差 [rad]

    // 数値安定性のための閾値
    static constexpr double COVARIANCE_SINGULARITY_THRESHOLD = 1e-12;  // 共分散行列の特異性チェック閾値
    static constexpr double CONDITION_NUMBER_WARNING_THRESHOLD = 1e12; // 条件数の警告閾値

    // ===================== 初期化 =====================

    void initializeParameters()
    {
        this->validateParameters();
    }

    void initializeWorkingMatrices()
    {
        this->F_temp_.setIdentity();
        this->G_temp_.setZero();
        this->H_temp_.setZero();
        this->Q_temp_.setZero();
    }

    void validateParameters() const
    {
        if (this->params_.position_noise <= 0 || this->params_.velocity_noise <= 0 ||
            this->params_.orientation_noise <= 0 || this->params_.angular_vel_noise <= 0)
        {
            throw std::invalid_argument("Process noise parameters must be positive");
        }
        if (this->params_.pos_noise <= 0 || this->params_.ori_noise <= 0)
        {
            throw std::invalid_argument("Measurement noise parameters must be positive");
        }
    }

    // ===================== 予測段階 =====================

    /**
     * @brief 予測段階（内部使用、dtはクラス内で管理）
     * @param dt 時間差分 [sec]
     */
    void predict(const double dt)
    {
        if (dt <= 0.0)
        {
            std::cerr << "Warning: Invalid dt = " << dt << " sec. Prediction skipped." << std::endl;
            return;
        }

        try
        {
            this->predictNominalState(dt);
            this->predictErrorState(dt);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in prediction: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * @brief 名目状態の予測（非線形積分）
     * @param dt 時間差分 [sec]
     */
    void predictNominalState(const double dt)
    {
        // 等速運動モデル
        this->nominal_state_.position += this->nominal_state_.velocity * dt;

        // 等角速度運動モデル
        if (this->nominal_state_.angular_velocity.norm() > EPSILON_ANGLE)
        {
            const Eigen::Quaterniond delta_q = this->axisAngleToQuaternion(this->nominal_state_.angular_velocity * dt);
            this->nominal_state_.quaternion = this->nominal_state_.quaternion * delta_q;
            this->nominal_state_.quaternion.normalize();
        }

        // 速度と角速度は定数（random walkモデルとして、プロセスノイズで不確実性が増加）
    }

    /**
     * @brief 誤差状態の予測（線形化）
     * @param dt 時間差分 [sec]
     */
    void predictErrorState(const double dt)
    {
        // 状態遷移行列
        this->computeStateTransitionMatrix(dt, this->F_temp_);

        // プロセスノイズ行列 G (ノイズが誤差状態にどのように影響するか)
        this->computeProcessNoiseMatrix(dt, this->G_temp_);
        // プロセスノイズ共分散 Q
        this->computeProcessNoiseCovariance(this->Q_temp_);

        // 共分散予測: P_k+1|k = F_k P_k|k F_k^T + G_k Q_k G_k^T
        this->error_covariance_ = this->F_temp_ * this->error_covariance_ * this->F_temp_.transpose() +
                                  this->G_temp_ * this->Q_temp_ * this->G_temp_.transpose();

        // 共分散行列の数値安定性チェック
        this->checkCovarianceHealth();
    }

    /**
     * @brief 状態遷移行列 F の計算（インプレース版）
     * @param dt 時間差分 [sec]
     * @param F 出力先の状態遷移行列
     */
    void computeStateTransitionMatrix(const double dt, StateTransition &F) const
    {
        F.setIdentity();

        // δp' = δp + δv * dt
        F.template block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

        // δv' = δv (等速モデル)

        // δθ' = δθ - [ω_n]× * δθ * dt - δω * dt
        // ここでの δθ は回転ベクトルの誤差
        const Eigen::Matrix3d omega_skew = this->skewSymmetric(this->nominal_state_.angular_velocity);
        F.template block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - omega_skew * dt;
        F.template block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

        // δω' = δω (定数モデル)
    }

    /**
     * @brief プロセスノイズ行列 G の計算（インプレース版）
     * @param dt 時間差分 [sec]
     * @param G 出力先のプロセスノイズ行列
     */
    void computeProcessNoiseMatrix(const double dt, ProcessNoise &G) const
    {
        G.setZero();

        // プロセスノイズの伝播モデル
        G.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt; // δp ← n_p (via velocity)
        G.template block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();      // δv ← n_v (direct)
        G.template block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * dt; // δθ ← n_θ (via angular velocity)
        G.template block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();      // δω ← n_ω (direct)
    }

    /**
     * @brief プロセスノイズ共分散 Q の計算（インプレース版）
     * @param Q 出力先のプロセスノイズ共分散行列
     */
    void computeProcessNoiseCovariance(ProcessNoise &Q) const
    {
        Q.setZero();

        // 各ノイズの分散 (標準偏差の二乗) を設定
        const double pos_var = this->params_.position_noise * this->params_.position_noise;
        const double vel_var = this->params_.velocity_noise * this->params_.velocity_noise;
        const double ori_var = this->params_.orientation_noise * this->params_.orientation_noise;
        const double ang_vel_var = this->params_.angular_vel_noise * this->params_.angular_vel_noise;

        Q.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * pos_var;
        Q.template block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * vel_var;
        Q.template block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * ori_var;
        Q.template block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * ang_vel_var;
    }

    // ===================== 観測更新 =====================

    /**
     * @brief 観測ヤコビアン H の計算（インプレース版）
     * @param H 出力先の観測ヤコビアン行列
     */
    void computeObservationJacobian(ObservationJacobian &H) const
    {
        H.setZero();

        // 位置観測: innovation_p = odom_pose - p_n = (p_n + δp) - p_n = δp
        H.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // d(innovation_p)/d(δp)

        // 姿勢観測: innovation_theta = axisAngle(q_n.inverse * odom_quat)
        // odom_quat = q_n * Exp(δθ) * Exp(error_on_measurement)
        // innovation_theta ≈ δθ (小角度近似)
        H.template block<3, 3>(3, 6) = Eigen::Matrix3d::Identity(); // d(innovation_theta)/d(δθ)
    }

    /**
     * @brief カルマン更新（数値安定性を考慮）
     * @param H 観測ヤコビアン
     * @param innovation 残差ベクトル
     * @param R 観測共分散行列
     */
    void performKalmanUpdate(const ObservationJacobian &H,
                             const ObservationVector &innovation,
                             const ObservationCovariance &R)
    {

        // イノベーション共分散: S = H P H^T + R
        const InnovationCovariance S = H * this->error_covariance_ * H.transpose() + R;

        // 数値安定性チェック
        if (S.determinant() < COVARIANCE_SINGULARITY_THRESHOLD)
        {
            std::cerr << "Warning: Innovation covariance matrix near singular (det="
                      << S.determinant() << "). Skipping update." << std::endl;
            return;
        }

        // カルマンゲイン: K = P H^T S^-1 (LDLT分解を使用して数値安定性向上)
        const Eigen::LDLT<InnovationCovariance> ldlt_S(S);
        if (ldlt_S.info() != Eigen::Success)
        {
            std::cerr << "Warning: LDLT decomposition failed. Skipping update." << std::endl;
            return;
        }

        const auto PHt = this->error_covariance_ * H.transpose();
        const KalmanGain K = PHt * ldlt_S.solve(InnovationCovariance::Identity());

        // 誤差状態更新: δx_k|k = K * innovation
        const ErrorStateVector delta_x_vec = K * innovation;
        this->error_state_.fromVector(delta_x_vec);

        // 共分散更新（Joseph form for numerical stability）: P_k|k = (I - K H) P_k|k-1 (I - K H)^T + K R K^T
        const ErrorCovariance I = ErrorCovariance::Identity();
        const ErrorCovariance IKH = I - K * H;
        this->error_covariance_ = IKH * this->error_covariance_ * IKH.transpose() +
                                  K * R * K.transpose();

        // 更新後の共分散行列の健全性チェック
        this->checkCovarianceHealth();
    }

    /**
     * @brief 誤差注入とリセット（例外安全性向上版）
     */
    void injectErrorAndReset()
    {
        // バックアップを作成（例外安全性のため）
        const NominalState backup = this->nominal_state_;

        try
        {
            // 誤差を名目状態に注入
            this->nominal_state_.position += this->error_state_.delta_position;
            this->nominal_state_.velocity += this->error_state_.delta_velocity;
            this->nominal_state_.angular_velocity += this->error_state_.delta_angular_velocity;

            // 姿勢の誤差注入: q_n_new = q_n * Exp(δθ)
            if (this->error_state_.delta_orientation.norm() > EPSILON_ANGLE)
            {
                const Eigen::Quaterniond delta_q = this->axisAngleToQuaternion(this->error_state_.delta_orientation);
                this->nominal_state_.quaternion = this->nominal_state_.quaternion * delta_q;
                this->nominal_state_.quaternion.normalize();
            }

            // 誤差状態をゼロにリセット（最後に実行）
            this->error_state_ = ErrorState{};
        }
        catch (...)
        {
            // エラー時にはロールバック
            this->nominal_state_ = backup;
            throw;
        }
    }

    /**
     * @brief 共分散行列の健全性チェック
     */
    void checkCovarianceHealth()
    {
        // 対称性チェック
        const ErrorCovariance symmetry_check = this->error_covariance_ - this->error_covariance_.transpose();
        if (symmetry_check.norm() > 1e-10)
        {
            std::cerr << "Warning: Covariance matrix not symmetric (asymmetry norm="
                      << symmetry_check.norm() << ")" << std::endl;
            // 対称化
            this->error_covariance_ = 0.5 * (this->error_covariance_ + this->error_covariance_.transpose());
        }

        // 正定値チェック（対角成分が正であることを確認）
        for (int i = 0; i < this->error_covariance_.rows(); ++i)
        {
            if (this->error_covariance_(i, i) <= 0)
            {
                std::cerr << "Warning: Covariance matrix not positive definite (diagonal element "
                          << i << " = " << this->error_covariance_(i, i) << ")" << std::endl;
                // 小さな正の値で置換
                this->error_covariance_(i, i) = INITIAL_COV_SMALL_VALUE;
            }
        }

        // 条件数チェック
        const double condition_number = this->getCovarianceConditionNumber();
        if (std::isfinite(condition_number) && condition_number > CONDITION_NUMBER_WARNING_THRESHOLD)
        {
            std::cerr << "Warning: Covariance matrix poorly conditioned (condition number="
                      << condition_number << ")" << std::endl;
        }
    }

    // ===================== ユーティリティ関数 =====================

    /**
     * @brief 軸角→四元数変換（Eigen::AngleAxis使用版）
     * @param axis_angle 軸角ベクトル (回転軸 * 回転角)
     * @return 対応する四元数
     */
    Eigen::Quaterniond axisAngleToQuaternion(const Eigen::Vector3d &axis_angle) const
    {
        const double angle = axis_angle.norm();
        if (angle < EPSILON_ANGLE)
        {
            return Eigen::Quaterniond::Identity();
        }

        // Eigen::AngleAxisdを直接使用（内部最適化済み）
        return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis_angle.normalized()));
    }

    /**
     * @brief 四元数→軸角変換（Eigen::AngleAxis使用版、数値安定性向上）
     * 最短回転を返す (角度は [0, pi] の範囲)
     * @param q 四元数
     * @return 対応する軸角ベクトル
     */
    Eigen::Vector3d quaternionToAxisAngle(const Eigen::Quaterniond &q) const
    {
        Eigen::Quaterniond q_normalized = q.normalized();

        // 最短回転の保証（四元数の符号の一意性を確保）
        // q と -q は同じ回転を表すが、AngleAxisdは[0,π]の角度を返すため
        if (q_normalized.w() < 0)
        {
            q_normalized.coeffs() *= -1;
        }

        // Eigen::AngleAxisdによる変換（内部で数値安定性が考慮されている）
        const Eigen::AngleAxisd angle_axis(q_normalized);
        const double angle = angle_axis.angle();

        // 極小回転の場合（Eigenが既に処理しているが、念のため）
        if (angle < EPSILON_ANGLE)
        {
            return Eigen::Vector3d::Zero();
        }

        return angle * angle_axis.axis();
    }

    /**
     * @brief 歪対称行列 (クロス積行列)
     * @param v ベクトル
     * @return 歪対称行列
     */
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) const
    {
        Eigen::Matrix3d skew;
        skew << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return skew;
    }

    /**
     * @brief デバッグ用状態ログ
     * @param stage ログのステージ名
     */
    void logState(const std::string &stage) const
    {
        if (this->enable_logging_)
        {
            std::cout << "[" << stage << "] "
                      << "pos=(" << this->nominal_state_.position.transpose() << ") "
                      << "vel=(" << this->nominal_state_.velocity.transpose() << ") "
                      << "pos_std=(" << this->getPositionUncertainty().transpose() << ")"
                      << std::endl;
        }
    }

public:
    // ===================== パラメータ設定 =====================

    /**
     * @brief プロセスノイズパラメータ設定（個別指定）
     */
    void setProcessNoise(const double pos_noise, const double vel_noise,
                         const double ori_noise, const double angular_vel_noise)
    {
        this->params_.position_noise = pos_noise;
        this->params_.velocity_noise = vel_noise;
        this->params_.orientation_noise = ori_noise;
        this->params_.angular_vel_noise = angular_vel_noise;
        this->validateParameters();
    }

    /**
     * @brief プロセスノイズ共分散行列の直接設定
     * @param Q プロセスノイズ共分散行列 (12x12)
     */
    void setProcessNoiseMatrix(const ProcessNoise &Q)
    {
        if (Q.rows() != ERROR_STATE_DIM || Q.cols() != ERROR_STATE_DIM)
        {
            throw std::invalid_argument("Process noise matrix must be 12x12");
        }

        // 行列が対称正定値かチェック
        const ProcessNoise symmetry_check = Q - Q.transpose();
        if (symmetry_check.norm() > 1e-10)
        {
            throw std::invalid_argument("Process noise matrix must be symmetric");
        }

        // 対角成分が正かチェック
        for (int i = 0; i < Q.rows(); ++i)
        {
            if (Q(i, i) <= 0)
            {
                throw std::invalid_argument("Process noise matrix must be positive definite");
            }
        }

        // パラメータを逆算して設定（個別設定との整合性を保つため）
        this->params_.position_noise = std::sqrt(Q(0, 0));
        this->params_.velocity_noise = std::sqrt(Q(3, 3));
        this->params_.orientation_noise = std::sqrt(Q(6, 6));
        this->params_.angular_vel_noise = std::sqrt(Q(9, 9));
    }

    /**
     * @brief 観測ノイズパラメータ設定（個別指定）
     */
    void setMeasurementNoise(const double pos_noise, const double ori_noise)
    {
        this->params_.pos_noise = pos_noise;
        this->params_.ori_noise = ori_noise;
        this->validateParameters();
    }

    /**
     * @brief 観測ノイズ共分散行列の直接設定
     * @param R 観測ノイズ共分散行列 (6x6)
     */
    void setMeasurementNoiseMatrix(const ObservationCovariance &R)
    {
        if (R.rows() != OBSERVATION_DIM || R.cols() != OBSERVATION_DIM)
        {
            throw std::invalid_argument("Measurement noise matrix must be 6x6");
        }

        // 行列が対称正定値かチェック
        const ObservationCovariance symmetry_check = R - R.transpose();
        if (symmetry_check.norm() > 1e-10)
        {
            throw std::invalid_argument("Measurement noise matrix must be symmetric");
        }

        // 対角成分が正かチェック
        for (int i = 0; i < R.rows(); ++i)
        {
            if (R(i, i) <= 0)
            {
                throw std::invalid_argument("Measurement noise matrix must be positive definite");
            }
        }

        // パラメータを逆算して設定（個別設定との整合性を保つため）
        this->params_.pos_noise = std::sqrt(R(0, 0));
        this->params_.ori_noise = std::sqrt(R(3, 3));
    }

    /**
     * @brief デフォルト観測共分散行列取得
     */
    ObservationCovariance getDefaultMeasurementCovariance() const
    {
        ObservationCovariance R = ObservationCovariance::Zero();
        const double pos_var = this->params_.pos_noise * this->params_.pos_noise;
        const double ori_var = this->params_.ori_noise * this->params_.ori_noise;

        R.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * pos_var;
        R.template block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * ori_var;
        return R;
    }

    /**
     * @brief 現在のプロセスノイズ共分散行列取得
     */
    ProcessNoise getProcessNoiseMatrix() const
    {
        ProcessNoise Q;
        this->computeProcessNoiseCovariance(Q);
        return Q;
    }

    /**
     * @brief 現在の観測ノイズ共分散行列取得
     */
    ObservationCovariance getMeasurementNoiseMatrix() const
    {
        return this->getDefaultMeasurementCovariance();
    }

    /**
     * @brief 初期共分散行列の直接設定
     * @param initial_covariance 初期誤差共分散行列 (12x12)
     */
    void setInitialCovariance(const ErrorCovariance &initial_covariance)
    {
        if (initial_covariance.rows() != ERROR_STATE_DIM || initial_covariance.cols() != ERROR_STATE_DIM)
        {
            throw std::invalid_argument("Initial covariance matrix must be 12x12");
        }

        // 行列が対称正定値かチェック
        const ErrorCovariance symmetry_check = initial_covariance - initial_covariance.transpose();
        if (symmetry_check.norm() > 1e-10)
        {
            throw std::invalid_argument("Initial covariance matrix must be symmetric");
        }

        // 対角成分が正かチェック
        for (int i = 0; i < initial_covariance.rows(); ++i)
        {
            if (initial_covariance(i, i) <= 0)
            {
                throw std::invalid_argument("Initial covariance matrix must be positive definite");
            }
        }

        this->error_covariance_ = initial_covariance;
        this->checkCovarianceHealth();
    }

    /**
     * @brief 初期姿勢設定
     */
    void setInitialPose(const Eigen::Isometry3d &initial_pose,
                        const ObservationCovariance &initial_covariance = ObservationCovariance::Zero(),
                        const double initial_timestamp = 0.0)
    {

        // 初期姿勢の設定（四元数の正規化を保証）
        this->nominal_state_.position = initial_pose.translation();
        this->nominal_state_.quaternion = Eigen::Quaterniond(initial_pose.rotation()).normalized();

        // 初期速度と角速度をゼロに設定
        this->nominal_state_.velocity = Eigen::Vector3d::Zero();
        this->nominal_state_.angular_velocity = Eigen::Vector3d::Zero();

        this->last_timestamp_ = initial_timestamp;

        // 初期共分散の設定
        if (initial_covariance.isZero(EPSILON_TIME))
        {
            // デフォルト初期共分散 (標準偏差の二乗として設定)
            this->error_covariance_ = ErrorCovariance::Identity() * INITIAL_COV_SMALL_VALUE;
            const double pos_var = INITIAL_POS_UNCERT_STD * INITIAL_POS_UNCERT_STD;
            const double ori_var = INITIAL_ORI_UNCERT_STD * INITIAL_ORI_UNCERT_STD;

            this->error_covariance_.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * pos_var;
            this->error_covariance_.template block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * ori_var;
        }
        else
        {
            this->error_covariance_.template block<3, 3>(0, 0) = initial_covariance.template block<3, 3>(0, 0);
            this->error_covariance_.template block<3, 3>(6, 6) = initial_covariance.template block<3, 3>(3, 3);
            // 速度と角速度の初期共分散はデフォルト小値を使用
            this->error_covariance_.template block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * INITIAL_COV_SMALL_VALUE;
            this->error_covariance_.template block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * INITIAL_COV_SMALL_VALUE;
        }

        // 共分散行列の健全性チェック
        this->checkCovarianceHealth();
    }

    /**
     * @brief ログ機能の有効/無効切り替え
     */
    void enableLogging(const bool enable)
    {
        this->enable_logging_ = enable;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
# IMU拡張ESKFの誤差状態設計

## 前提
既存実装では、推定する公称状態が以下の 12 次元ベクトルで構成されています。

- 位置 $\mathbf{p}$（世界座標系）
- 速度 $\mathbf{v}$（世界座標系）
- 姿勢 $\mathbf{q}$（機体系→世界系のクォータニオン）
- 角速度 $\boldsymbol{\omega}$（機体系）

観測は GNSS や VIO などから得られる位置および姿勢であり、IMU（加速度計・ジャイロ）入力は未使用です。IMU を組み込むには、センサバイアスや IMU ノイズを誤差状態で扱えるよう、状態設計を拡張する必要があります。

## 公称状態の拡張
IMU をプロパゲーションに利用する場合、公称状態として以下の 16 次元ベクトルを想定するのが一般的です。

- 位置 $\mathbf{p} \in \mathbb{R}^3$
- 速度 $\mathbf{v} \in \mathbb{R}^3$（世界座標系）
- 姿勢 $\mathbf{q} \in \mathrm{SO}(3)$（クォータニオン）
- ジャイロバイアス $\mathbf{b}_g \in \mathbb{R}^3$
- 加速度バイアス $\mathbf{b}_a \in \mathbb{R}^3$
- （任意）重力ベクトル $\mathbf{g} \in \mathbb{R}^3$ ※既知とみなす場合は状態から除外

角速度そのものは IMU 入力として得られるため、公称状態に保持する必要はありません。重力は既知値を使用するか、推定が必要であれば状態に含めます（本設計では既知値を前提とします）。したがって、公称状態は 15 次元となります。

公称状態ベクトルを次のように定義します。
$$
\mathbf{x} = \begin{bmatrix}
\mathbf{p} & \mathbf{v} & \mathbf{q} & \mathbf{b}_g & \mathbf{b}_a
\end{bmatrix}.
$$
姿勢は単位クォータニオンで表すため制約付き状態になります。

## 誤差状態ベクトル
ESKF では公称状態と独立した誤差状態を線形空間で扱います。上記の公称状態に対して、誤差状態は以下の 15 次元ベクトルとします。

- 位置誤差 $\delta \mathbf{p}$
- 速度誤差 $\delta \mathbf{v}$
- 姿勢誤差 $\delta \boldsymbol{\theta}$（小回転ベクトル）
- ジャイロバイアス誤差 $\delta \mathbf{b}_g$
- 加速度バイアス誤差 $\delta \mathbf{b}_a$

$$
\delta \mathbf{x} = \begin{bmatrix}
\delta \mathbf{p} & \delta \mathbf{v} & \delta \boldsymbol{\theta} & \delta \mathbf{b}_g & \delta \mathbf{b}_a
\end{bmatrix}^T \in \mathbb{R}^{15}.
$$

### 誤差ダイナミクス
IMU 入力（角速度 $\tilde{\boldsymbol{\omega}}$、加速度 $\tilde{\mathbf{f}}$）を用いた誤差ダイナミクスは次のようにモデル化できます。

- $\dot{\delta \mathbf{p}} = \delta \mathbf{v}$
- $\dot{\delta \mathbf{v}} = -\mathbf{R}(\hat{\mathbf{q}})\,[\tilde{\mathbf{f}}-\hat{\mathbf{b}}_a]_\times\,\delta \boldsymbol{\theta} - \mathbf{R}(\hat{\mathbf{q}})\,\delta \mathbf{b}_a - \mathbf{R}(\hat{\mathbf{q}})\,\mathbf{n}_a$
- $\dot{\delta \boldsymbol{\theta}} = -\mathrm{skew}(\tilde{\boldsymbol{\omega}}-\hat{\mathbf{b}}_g)\,\delta \boldsymbol{\theta} - \delta \mathbf{b}_g - \mathbf{n}_g$
- $\dot{\delta \mathbf{b}}_g = \mathbf{n}_{bg}$
- $\dot{\delta \mathbf{b}}_a = \mathbf{n}_{ba}$

ここで、$\mathbf{n}_g$ は角速度ノイズ、$\mathbf{n}_a$ は加速度ノイズ、$\mathbf{n}_{bg}, \mathbf{n}_{ba}$ はそれぞれジャイロ・加速度バイアスのランダムウォークノイズです。$\mathbf{R}(\hat{\mathbf{q}})$ は公称姿勢から得られる回転行列、$[\cdot]_\times$ および $\mathrm{skew}(\cdot)$ は歪対称行列演算子を表します。

### プロセスノイズ
プロセスノイズベクトルは
$$
\mathbf{w} = \begin{bmatrix}
\mathbf{n}_g & \mathbf{n}_a & \mathbf{n}_{bg} & \mathbf{n}_{ba}
\end{bmatrix}^T
$$
とし、共分散行列は角速度・加速度の測定ノイズ密度、およびバイアスランダムウォーク密度を対角成分に配置した $12\times12$ 行列とします。

## 観測モデル
位置・姿勢の外部観測（GNSS/VIO 等）に対して、観測ベクトルは
$$
\mathbf{z} = \begin{bmatrix}
\mathbf{p}_{\text{meas}} & \mathbf{q}_{\text{meas}}
\end{bmatrix}
$$
となります。線形化した観測モデルでは、位置には $\delta \mathbf{p}$、姿勢には $\delta \boldsymbol{\theta}$ が直接掛かるため、観測行列 $\mathbf{H}$ の構造は以下の通りです。

$$
\mathbf{H} =
\begin{bmatrix}
\mathbf{I}_{3} & \mathbf{0}_{3} & \mathbf{0}_{3} & \mathbf{0}_{3} & \mathbf{0}_{3} \\
\mathbf{0}_{3} & \mathbf{0}_{3} & \mathbf{I}_{3} & \mathbf{0}_{3} & \mathbf{0}_{3}
\end{bmatrix}.
$$

バイアス誤差は直接観測されないため、観測行列には現れません。

## まとめ
- 公称状態は $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \mathbf{b}_g, \mathbf{b}_a]$ の 15 次元。
- 誤差状態は $[\delta \mathbf{p}, \delta \mathbf{v}, \delta \boldsymbol{\theta}, \delta \mathbf{b}_g, \delta \mathbf{b}_a]$ の 15 次元。
- プロセスノイズにはジャイロ／加速度測定ノイズと、それぞれのバイアスランダムウォークを含める。
- 観測行列は位置・姿勢に対してそれぞれ単位行列を配置し、バイアス誤差は間接的に推定する。

この設計により、IMU 入力を用いた予測ステップでバイアスを補償しながら位置・姿勢を推定でき、外部観測による補正でバイアスも推定可能になります。

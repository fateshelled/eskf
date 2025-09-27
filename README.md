# ESKF
Error State Kalman filter

```bash
sudo apt install freeglut3-dev libglu1-mesa-dev libeigen3-dev

git clone https://github.com/fateshelled/eskf

cd eskf/eskf
g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include/eigen3 \
    main.cpp \
    -lGL -lGLU -lglut

./a.out
```

## Square-root ESKF

`SquareRootESKF` は既存の誤差状態カルマンフィルタをベースに、共分散のコレスキー因子を直接管理する平方根カルマンフィルタ実装です。実装は `eskf/square_root_eskf.hpp` にまとまっています。

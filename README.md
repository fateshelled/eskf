# ESKF
**Error-State Kalman filter** and **Square-Root Error-State Kalman filter** C++ implement

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

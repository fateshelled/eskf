# ESKF
Error State Kalman filter

```bash
sudo apt insatll freeglut3-dev libglu1-mesa-dev

git clone https://github.com/fateshelled/eskf

cd eskf/eskf
g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include/eigen3 \
    main.cpp \
    -lGL -lGLU -lglut

./a.out
```

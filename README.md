## Cuda pathtracer

### Dependencies

- glfw3 (apt)
- nvidia-cuda-toolkit (apt)
- cmake 3.16  (not sure if default apt or ppa needed)

### How to build

Inside the directory of the project you need to initialize the cmake cache the first time you build.

`cmake .`

Then you can build with `cmake --build .` which creates the `pathtracer` executable



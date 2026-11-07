#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "vec3.cuh"

class sphere {
public:
  vec3 center;
  float radius;
  vec3 color;
  bool isFloor;
  volatile bool padding1;
  volatile bool padding2;
  volatile bool padding3;
  __host__ __device__ sphere()
      : center{vec3(0, 0, 0)}, radius{0.0f}, color{vec3(0, 0, 0)},
        isFloor{false} {}
  __host__ __device__ sphere(vec3 center, float radius, vec3 color,
                             bool isFloor)
      : center{center}, radius{radius}, color{color}, isFloor{isFloor} {}
};

class spheres {
public:
  vec3 center[501];
  float radius[501];
  vec3 color[501];
  bool isFloor[501];

  spheres() {}
  __host__ __device__ spheres(int count) {
    for (int i = 0; i < 501; i++) {
      center[i] = vec3(0, 0, 0);
      radius[i] = 0.0f;
      color[i] = vec3(0, 0, 0);
      isFloor[i] = false;
    }
  }

  __host__ __device__ spheres(const sphere *spheres, int count) {
    for (int i = 0; i < count; i++) {
      center[i] = spheres[i].center;
      radius[i] = spheres[i].radius;
      color[i] = spheres[i].color;
      isFloor[i] = spheres[i].isFloor;
    }
  }
};
#endif

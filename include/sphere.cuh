#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "vec3.cuh"

class sphere {
public:
  vec3 center;
  float radius;
  vec3 color;
  bool isFloor;
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

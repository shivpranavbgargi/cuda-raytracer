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
#endif

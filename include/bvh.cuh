#ifndef BVH_CUH
#define BVH_CUH

#include "vec3.cuh"

class bvhPrimitive {
public:
  vec3 aabbMin;
  vec3 aabbMax;
  vec3 centroid;
  unsigned int sphereIdx;

  __host__ bvhPrimitive() : aabbMin{}, aabbMax{}, centroid{}, sphereIdx{0} {}

  __host__ bvhPrimitive(const vec3 aabbMin, const vec3 aabbMax,
                        const vec3 centroid, unsigned int sphereIdx)
      : aabbMin{aabbMin}, aabbMax{aabbMax}, centroid{centroid},
        sphereIdx{sphereIdx} {}
};

class bvhNode {
public:
  vec3 aabbMin;
  vec3 aabbMax;
  bvhNode *left;
  bvhNode *right;
  int sphereIdx;
  bool isLeaf;

  __host__ bvhNode()
      : aabbMin{}, aabbMax{}, left(nullptr), right(nullptr), sphereIdx(-1),
        isLeaf(false) {}
};

class flatBvhNode {
public:
  vec3 aabbMin;
  vec3 aabbMax;
  int left;
  int right;
  int sphereIdx;
};

class bvh {
public:
  bvhNode *root;
};

#endif

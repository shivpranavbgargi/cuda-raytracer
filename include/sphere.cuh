#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "vec3.cuh"
#include <vector>

class sphere {
public:
  vec3 center;
  double radius;
  vec3 color;
  bool isFloor;
  __host__ __device__ sphere(vec3 center, double radius, vec3 color,
                             bool isFloor)
      : center{center}, radius{radius}, color{color}, isFloor{isFloor} {}
};

// __host__ __device__ sphere scene;

// __host__ __device__ std::vector<sphere> scene = {
//     { vec3(0, -1000.5f, -1), 1000.0f, vec3(0.5f, 0.5f, 0.5f), true  }, //
//     floor { vec3(0,     0.0f, -1),    0.5f, vec3(0.9f, 0.2f, 0.2f), false },
//     // red { vec3(-1.1f, 0.0f, -1),    0.5f, vec3(0.2f, 0.7f, 0.3f), false },
//     // green { vec3( 1.1f, 0.0f, -1),    0.5f, vec3(0.2f, 0.4f, 0.9f), false
//     }, // blue
// };

__host__ __device__ inline double hitSphere(const sphere &s, const vec3 origin,
                                            vec3 dir) {
  vec3 oc = origin - s.center;
  double a = dot(dir, dir);
  double b = 2.0f * dot(oc, dir);
  double c = dot(oc, oc) - s.radius * s.radius;
  double disc = b * b - 4 * a * c;
  if (disc < 0)
    return -1.0f;                              // no real solution -> miss
  double t = (-b - std::sqrt(disc)) / (2 * a); // nearest intersection
  return (t > 0.001f) ? t : -1.0f;
}

__host__ __device__ inline int closestHit(vec3 origin, vec3 dir, double &tHit,
                                          sphere *scene, int size) {
  int best = -1;
  double closest = 1e30f;
  // for (int i = 0; i < (int)scene.size(); i++) {
  for (int i = 0; i < size; i++) {
    double t = hitSphere(scene[i], origin, dir);
    if (t > 0 && t < closest) {
      closest = t;
      best = i;
    }
  }
  tHit = closest;
  return best;
}

__host__ __device__ inline vec3 rayColor(vec3 origin, vec3 dir, sphere *scene) {
  // TRY IT: move the light around and see how shading/shadows change.
  // x = right, y = up, z = toward the camera.
  vec3 lightPos(4, 6, 2);
  double tHit;
  int hit = closestHit(origin, dir, tHit, scene, 4);

  if (hit < 0) {
    // Miss: draw a simple sky gradient (white near horizon -> blue up top).
    double t = 0.5f * (dir.y() + 1.0f);
    return vec3(1, 1, 1) * (1 - t) + vec3(0.4f, 0.6f, 1.0f) * t;
  }

  // The point we hit, and the surface normal there.
  vec3 p = origin + dir * tHit;
  vec3 n = normalize(p - scene[hit].center);

  // Base color: checkerboard for the floor, solid color otherwise.
  vec3 base = scene[hit].color;
  if (scene[hit].isFloor) {
    int cx = (int)std::floor(p.x()), cz = (int)std::floor(p.z());
    bool white = ((cx + cz) & 1) == 0;
    base = white ? vec3(0.9f, 0.9f, 0.9f) : vec3(0.2f, 0.2f, 0.2f);
  }

  // Diffuse lighting: how directly does the surface face the light?
  vec3 toLight = normalize(lightPos - p);
  double diffuse = fmaxf(0.0f, dot(n, toLight));

  // Hard shadow: shoot a ray toward the light; if it hits something, we're
  // shadowed.
  double shadowT;
  int blocked = closestHit(p + n * 0.001f, toLight, shadowT, scene, 4);
  double shade = (blocked >= 0) ? 0.3f : 1.0f;

  double brightness = 0.15f + diffuse * shade; // 0.15 = a little ambient light
  return base * brightness;
}

#endif

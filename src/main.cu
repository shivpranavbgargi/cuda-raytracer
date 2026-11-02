#include "sphere.cuh"
#include "vec3.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cuda.h>
#include <iostream>

using namespace std;
namespace cg = cooperative_groups;

#define N 201

__global__ void cuda_gen_gradient(int W, int H, int *out) {
  const float ALPHA = 255.999; // scaling factor

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int outIdx = 3 * ((y * W) + x);
  if (y < H && x < W) {
    auto r = float(x) / (W - 1);
    auto g = float(y) / (H - 1);
    auto b = 0.0;
    auto pixel =
        vec3(r, g, b); // TODO: Check if malloc outside and fill here is faster?
    pixel = ALPHA * pixel;
    out[outIdx] = pixel.x();
    out[outIdx + 1] = pixel.y();
    out[outIdx + 2] = pixel.z();
  }
}

void gen_ppm(int W, int H, int *dPpm, int *hPpm) {
  // TODO: Test perf for different TX/TY
  const int TX = 8;
  const int TY = 8;
  dim3 block(TX, TY, 1);
  dim3 grid((W + TX - 1) / TX, (H + TY - 1) / TY, 1);

  cuda_gen_gradient<<<grid, block>>>(W, H, dPpm);
  cudaDeviceSynchronize(); // redundant but good practice
  cudaMemcpy(hPpm, dPpm, 3 * H * W * sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "P3\n" << W << ' ' << H << "\n255\n";
  for (int i = 0; i < (W * H); i++) {
    int base_idx = 3 * i;
    std::cout << hPpm[base_idx] << ' ' << hPpm[base_idx + 1] << ' '
              << hPpm[base_idx + 2] << '\n';
  }
}

__host__ __device__ inline float hitSphere(const sphere &s, const vec3 origin,
                                           vec3 dir) {
  vec3 oc = origin - s.center;
  float a = dot(dir, dir);
  float b = 2.0f * dot(oc, dir);
  float c = dot(oc, oc) - s.radius * s.radius;
  float disc = b * b - 4 * a * c;
  if (disc < 0)
    return -1.0f;                             // no real solution -> miss
  float t = (-b - std::sqrt(disc)) / (2 * a); // nearest intersection
  return (t > 0.001f) ? t : -1.0f;
}

__host__ __device__ inline int closestHit(vec3 origin, vec3 dir, float &tHit,
                                          sphere *scene, int size) {
  int best = -1;
  float closest = 1e30f;
  for (int i = 0; i < size; i++) {
    float t = hitSphere(scene[i], origin, dir);
    if (t > 0 && t < closest) {
      closest = t;
      best = i;
    }
  }
  tHit = closest;
  return best;
}

__device__ inline vec3 rayColor(vec3 origin, vec3 dir, sphere *scene,
                                unsigned int bounce) {
  // TRY IT: move the light around and see how shading/shadows change.
  // x = right, y = up, z = toward the camera.
  vec3 lightPos(4, 6, 2);
  float k = 0.8;
  float tHit;
  vec3 color = vec3(0, 0, 0);

  vec3 p = vec3(0, 0, 0);
  vec3 n = vec3(0, 0, 0);

  for (int i = 0; i < bounce; i++) {
    int hit = closestHit(origin, dir, tHit, scene, N);

    if (hit < 0) {
      // Miss: draw a simple sky gradient (white near horizon -> blue up top).
      float t = 0.5f * (dir.y() + 1.0f);
      if (i == 0) {
        color = color + (vec3(1, 1, 1) * (1 - t) + vec3(0.4f, 0.6f, 1.0f) * t);
      } else {
        color =
            color + k * (vec3(1, 1, 1) * (1 - t) + vec3(0.4f, 0.6f, 1.0f) * t);
      }
      break;
    }

    p = origin + dir * tHit;
    n = normalize(p - scene[hit].center);

    // Base color: checkerboard for the floor, solid color otherwise.
    vec3 base = scene[hit].color;
    if (scene[hit].isFloor) {
      int cx = (int)floor(p.x()), cz = (int)floor(p.z());
      bool white = ((cx + cz) & 1) == 0;
      base = white ? vec3(0.9f, 0.9f, 0.9f) : vec3(0.2f, 0.2f, 0.2f);
    }

    // Diffuse lighting: how directly does the surface face the light?
    vec3 toLight = normalize(lightPos - p);
    float diffuse = fmaxf(0.0f, dot(n, toLight));

    // Hard shadow: shoot a ray toward the light; if it hits something, we're
    // shadowed.
    float shadowT;
    int blocked = closestHit(p + n * 0.001f, toLight, shadowT, scene, N);
    float shade = (blocked >= 0) ? 0.3f : 1.0f;

    float brightness = 0.15f + diffuse * shade; // 0.15 = a little ambient light
    color = color + k * base * brightness;

    k = 0.2;
    vec3 rDir = normalize(dir - (2 * dot(dir, n) * n));
    origin = p + rDir * 0.01f;
    dir = rDir;
  }
  return color;
}

__global__ void gen_scene_kernel(int W, int H, sphere *scene, int *out) {

  int px = blockDim.x * blockIdx.x + threadIdx.x;
  int py = blockDim.y * blockIdx.y + threadIdx.y;
  int outIdx = 3 * ((py * W) + px);
#ifdef DEBUG
  for (int i = 0; i < 4; i++) {
    printf(
        "center: %lf %lf %lf; radius: %lf; color: %lf %lf %lf; isFloor: %d\n",
        scene[i].center.x(), scene[i].center.y(), scene[i].center.z(),
        scene[i].radius, scene[i].color.x(), scene[i].color.y(),
        scene[i].color.z(), (int)scene[i].isFloor);
  }
#endif
  if (py < H && px < W) {
    float u = (2.0f * (px + 0.5f) / W - 1.0f) * (float)W / H;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);

    vec3 cameraPos = vec3(0, 1, 3);
    vec3 dir = normalize(vec3(u, v, -1.5f));

    vec3 color = rayColor(cameraPos, dir, scene, 2);

    auto r = (int)(fminf(color.x(), 1.0f) * 255);
    auto g = (int)(fminf(color.y(), 1.0f) * 255);
    auto b = (int)(fminf(color.z(), 1.0f) * 255);

    auto pixel = vec3(r, g, b);
    out[outIdx] = pixel.x();
    out[outIdx + 1] = pixel.y();
    out[outIdx + 2] = pixel.z();
  }
}

sphere hScene[N] = {
    {vec3(0, -1000.5f, -1), 1000.0f, vec3(0.5f, 0.5f, 0.5f), true},
    {vec3(-1.0274f, -0.2221f, -0.4252f), 0.2779f, vec3(0.329f, 0.739f, 0.691f),
     false},
    {vec3(-0.4662f, -0.1716f, -1.4780f), 0.3284f, vec3(0.174f, 0.325f, 0.554f),
     false},
    {vec3(1.2751f, -0.3447f, -3.3802f), 0.1553f, vec3(0.586f, 0.326f, 0.621f),
     false},
    {vec3(1.2683f, -0.1881f, -0.1105f), 0.3119f, vec3(0.709f, 0.422f, 0.274f),
     false},
    {vec3(-4.7362f, -0.1586f, -5.7221f), 0.3414f, vec3(0.227f, 0.828f, 0.633f),
     false},
    {vec3(0.7441f, -0.1886f, -12.4054f), 0.3114f, vec3(0.928f, 0.453f, 0.592f),
     false},
    {vec3(6.5179f, -0.1841f, -10.5148f), 0.3159f, vec3(0.612f, 0.714f, 0.187f),
     false},
    {vec3(-4.4372f, -0.3044f, -4.9196f), 0.1956f, vec3(0.336f, 0.231f, 0.372f),
     false},
    {vec3(-1.5928f, -0.2229f, -6.2021f), 0.2771f, vec3(0.318f, 0.364f, 0.899f),
     false},
    {vec3(-5.8560f, -0.2204f, -10.3552f), 0.2796f, vec3(0.733f, 0.281f, 0.454f),
     false},
    {vec3(1.0539f, -0.1521f, -10.8800f), 0.3479f, vec3(0.698f, 0.824f, 0.771f),
     false},
    {vec3(3.8898f, -0.2965f, -3.5867f), 0.2035f, vec3(0.851f, 0.402f, 0.674f),
     false},
    {vec3(-1.0176f, -0.2709f, -15.5473f), 0.2291f, vec3(0.362f, 0.347f, 0.599f),
     false},
    {vec3(-2.7448f, -0.2481f, -1.5455f), 0.2519f, vec3(0.238f, 0.652f, 0.784f),
     false},
    {vec3(7.5341f, -0.1508f, -8.9949f), 0.3492f, vec3(0.839f, 0.159f, 0.727f),
     false},
    {vec3(-3.7707f, -0.2137f, -9.1285f), 0.2863f, vec3(0.663f, 0.239f, 0.498f),
     false},
    {vec3(9.6293f, -0.2593f, -16.2149f), 0.2407f, vec3(0.361f, 0.550f, 0.293f),
     false},
    {vec3(-4.7833f, -0.1675f, -14.7988f), 0.3325f, vec3(0.661f, 0.637f, 0.272f),
     false},
    {vec3(4.5210f, -0.1975f, -9.1694f), 0.3025f, vec3(0.574f, 0.150f, 0.409f),
     false},
    {vec3(-10.5510f, -0.1744f, -16.0981f), 0.3256f,
     vec3(0.539f, 0.205f, 0.758f), false},
    {vec3(3.7273f, -0.2400f, -4.5060f), 0.2600f, vec3(0.489f, 0.319f, 0.581f),
     false},
    {vec3(-1.6116f, -0.2040f, -3.4196f), 0.2960f, vec3(0.946f, 0.670f, 0.500f),
     false},
    {vec3(-1.8563f, -0.2465f, -2.0571f), 0.2535f, vec3(0.420f, 0.621f, 0.334f),
     false},
    {vec3(0.7354f, -0.3060f, -1.2069f), 0.1940f, vec3(0.333f, 0.874f, 0.838f),
     false},
    {vec3(1.5875f, -0.3358f, -4.0461f), 0.1642f, vec3(0.321f, 0.256f, 0.898f),
     false},
    {vec3(4.1879f, -0.2358f, -8.0354f), 0.2642f, vec3(0.796f, 0.302f, 0.228f),
     false},
    {vec3(-0.4485f, -0.2638f, -7.2008f), 0.2362f, vec3(0.733f, 0.689f, 0.937f),
     false},
    {vec3(-2.1093f, -0.3303f, -6.8446f), 0.1697f, vec3(0.839f, 0.349f, 0.302f),
     false},
    {vec3(-3.0035f, -0.2603f, -7.1720f), 0.2397f, vec3(0.350f, 0.889f, 0.505f),
     false},
    {vec3(-7.4036f, -0.1777f, -9.3555f), 0.3223f, vec3(0.949f, 0.819f, 0.925f),
     false},
    {vec3(-7.7540f, -0.1647f, -14.4278f), 0.3353f, vec3(0.539f, 0.321f, 0.471f),
     false},
    {vec3(6.1101f, -0.3383f, -6.4425f), 0.1617f, vec3(0.362f, 0.777f, 0.514f),
     false},
    {vec3(-7.0021f, -0.2388f, -12.2129f), 0.2612f, vec3(0.387f, 0.925f, 0.613f),
     false},
    {vec3(-9.2792f, -0.2416f, -12.7156f), 0.2584f, vec3(0.617f, 0.552f, 0.832f),
     false},
    {vec3(3.0640f, -0.3128f, -10.1156f), 0.1872f, vec3(0.338f, 0.246f, 0.862f),
     false},
    {vec3(2.0863f, -0.3008f, -10.1068f), 0.1992f, vec3(0.485f, 0.617f, 0.568f),
     false},
    {vec3(2.2270f, -0.3023f, -6.7284f), 0.1977f, vec3(0.390f, 0.403f, 0.751f),
     false},
    {vec3(-1.6237f, -0.1508f, -1.2454f), 0.3492f, vec3(0.362f, 0.897f, 0.855f),
     false},
    {vec3(2.2276f, -0.1833f, -11.9602f), 0.3167f, vec3(0.940f, 0.673f, 0.156f),
     false},
    {vec3(1.7623f, -0.1866f, -5.0894f), 0.3134f, vec3(0.901f, 0.257f, 0.242f),
     false},
    {vec3(-6.0069f, -0.2290f, -12.1994f), 0.2710f, vec3(0.657f, 0.361f, 0.541f),
     false},
    {vec3(-9.4498f, -0.1689f, -14.3838f), 0.3311f, vec3(0.489f, 0.371f, 0.153f),
     false},
    {vec3(-4.3898f, -0.1958f, -10.8309f), 0.3042f, vec3(0.743f, 0.591f, 0.492f),
     false},
    {vec3(2.1858f, -0.3481f, -1.2791f), 0.1519f, vec3(0.873f, 0.586f, 0.818f),
     false},
    {vec3(-2.7408f, -0.2335f, -2.5176f), 0.2665f, vec3(0.397f, 0.869f, 0.787f),
     false},
    {vec3(-7.0671f, -0.1779f, -15.2817f), 0.3221f, vec3(0.350f, 0.232f, 0.774f),
     false},
    {vec3(9.1433f, -0.3191f, -15.8080f), 0.1809f, vec3(0.931f, 0.799f, 0.855f),
     false},
    {vec3(-3.4730f, -0.3450f, -12.5216f), 0.1550f, vec3(0.895f, 0.792f, 0.841f),
     false},
    {vec3(2.8874f, -0.1879f, -4.5357f), 0.3121f, vec3(0.236f, 0.848f, 0.837f),
     false},
    {vec3(-0.8935f, -0.3055f, -13.8820f), 0.1945f, vec3(0.394f, 0.786f, 0.332f),
     false},
    {vec3(-5.7242f, -0.1771f, -16.4371f), 0.3229f, vec3(0.663f, 0.470f, 0.935f),
     false},
    {vec3(-9.7278f, -0.2428f, -15.9670f), 0.2572f, vec3(0.926f, 0.293f, 0.920f),
     false},
    {vec3(0.9750f, -0.2477f, -6.5483f), 0.2523f, vec3(0.354f, 0.717f, 0.151f),
     false},
    {vec3(3.5558f, -0.1649f, -9.1537f), 0.3351f, vec3(0.744f, 0.687f, 0.441f),
     false},
    {vec3(-3.2357f, -0.3360f, -11.2920f), 0.1640f, vec3(0.401f, 0.828f, 0.726f),
     false},
    {vec3(-1.0086f, -0.2899f, -5.2578f), 0.2101f, vec3(0.472f, 0.387f, 0.252f),
     false},
    {vec3(4.4888f, -0.2659f, -15.9862f), 0.2341f, vec3(0.872f, 0.642f, 0.391f),
     false},
    {vec3(-4.0146f, -0.2570f, -7.5167f), 0.2430f, vec3(0.529f, 0.871f, 0.787f),
     false},
    {vec3(3.6929f, -0.2234f, -5.6982f), 0.2766f, vec3(0.751f, 0.688f, 0.330f),
     false},
    {vec3(0.4618f, -0.3148f, -16.2854f), 0.1852f, vec3(0.190f, 0.349f, 0.829f),
     false},
    {vec3(3.7144f, -0.2587f, -13.6241f), 0.2413f, vec3(0.940f, 0.626f, 0.910f),
     false},
    {vec3(3.9221f, -0.1717f, -10.4151f), 0.3283f, vec3(0.554f, 0.814f, 0.588f),
     false},
    {vec3(-0.5282f, -0.1706f, -12.6421f), 0.3294f, vec3(0.357f, 0.348f, 0.660f),
     false},
    {vec3(2.0047f, -0.1968f, -8.8621f), 0.3032f, vec3(0.370f, 0.212f, 0.379f),
     false},
    {vec3(0.4516f, -0.2957f, -5.4351f), 0.2043f, vec3(0.261f, 0.335f, 0.705f),
     false},
    {vec3(2.0608f, -0.2660f, -15.3823f), 0.2340f, vec3(0.706f, 0.835f, 0.762f),
     false},
    {vec3(10.5852f, -0.1993f, -14.5086f), 0.3007f, vec3(0.485f, 0.748f, 0.587f),
     false},
    {vec3(-2.5249f, -0.2293f, -3.7492f), 0.2707f, vec3(0.499f, 0.173f, 0.419f),
     false},
    {vec3(-4.4095f, -0.2142f, -6.8734f), 0.2858f, vec3(0.524f, 0.252f, 0.648f),
     false},
    {vec3(-6.7647f, -0.3446f, -10.9267f), 0.1554f, vec3(0.519f, 0.190f, 0.453f),
     false},
    {vec3(2.9802f, -0.3077f, -5.5564f), 0.1923f, vec3(0.453f, 0.752f, 0.816f),
     false},
    {vec3(-4.0007f, -0.2421f, -16.9984f), 0.2579f, vec3(0.670f, 0.775f, 0.671f),
     false},
    {vec3(-7.6737f, -0.1992f, -16.1434f), 0.3008f, vec3(0.166f, 0.272f, 0.251f),
     false},
    {vec3(-4.7335f, -0.2161f, -9.5875f), 0.2839f, vec3(0.710f, 0.764f, 0.284f),
     false},
    {vec3(-8.0767f, -0.2286f, -12.7147f), 0.2714f, vec3(0.805f, 0.922f, 0.236f),
     false},
    {vec3(7.5347f, -0.3296f, -13.1322f), 0.1704f, vec3(0.630f, 0.247f, 0.937f),
     false},
    {vec3(-2.4559f, -0.2759f, -8.6013f), 0.2241f, vec3(0.830f, 0.808f, 0.234f),
     false},
    {vec3(3.2430f, -0.2085f, -7.4033f), 0.2915f, vec3(0.922f, 0.366f, 0.797f),
     false},
    {vec3(-0.9638f, -0.2424f, -8.2195f), 0.2576f, vec3(0.735f, 0.365f, 0.831f),
     false},
    {vec3(1.6035f, -0.3012f, -7.9000f), 0.1988f, vec3(0.453f, 0.173f, 0.831f),
     false},
    {vec3(2.6233f, -0.3136f, -3.6060f), 0.1864f, vec3(0.422f, 0.854f, 0.711f),
     false},
    {vec3(-8.9419f, -0.2518f, -13.4799f), 0.2482f, vec3(0.327f, 0.703f, 0.395f),
     false},
    {vec3(0.4554f, -0.2337f, -8.0454f), 0.2663f, vec3(0.490f, 0.747f, 0.415f),
     false},
    {vec3(-2.5210f, -0.2094f, -4.6056f), 0.2906f, vec3(0.247f, 0.304f, 0.246f),
     false},
    {vec3(-6.6988f, -0.2428f, -12.9572f), 0.2572f, vec3(0.323f, 0.537f, 0.730f),
     false},
    {vec3(0.1475f, -0.3141f, -0.2405f), 0.1859f, vec3(0.369f, 0.929f, 0.593f),
     false},
    {vec3(2.5285f, -0.2105f, -2.1468f), 0.2895f, vec3(0.543f, 0.848f, 0.609f),
     false},
    {vec3(-0.5641f, -0.3397f, -15.9981f), 0.1603f, vec3(0.808f, 0.471f, 0.209f),
     false},
    {vec3(2.2675f, -0.3263f, -12.9955f), 0.1737f, vec3(0.783f, 0.331f, 0.568f),
     false},
    {vec3(5.0549f, -0.2599f, -7.5263f), 0.2401f, vec3(0.942f, 0.394f, 0.647f),
     false},
    {vec3(9.2988f, -0.2281f, -12.5815f), 0.2719f, vec3(0.316f, 0.319f, 0.678f),
     false},
    {vec3(-3.3739f, -0.3186f, -2.9548f), 0.1814f, vec3(0.152f, 0.510f, 0.625f),
     false},
    {vec3(-1.8837f, -0.2178f, -15.8724f), 0.2822f, vec3(0.586f, 0.668f, 0.877f),
     false},
    {vec3(1.4517f, -0.2885f, -12.7323f), 0.2115f, vec3(0.381f, 0.249f, 0.701f),
     false},
    {vec3(-2.8223f, -0.2636f, -5.4795f), 0.2364f, vec3(0.223f, 0.920f, 0.819f),
     false},
    {vec3(-6.1910f, -0.2230f, -8.3593f), 0.2770f, vec3(0.428f, 0.417f, 0.686f),
     false},
    {vec3(7.9722f, -0.2924f, -16.0683f), 0.2076f, vec3(0.590f, 0.514f, 0.402f),
     false},
    {vec3(-2.4906f, -0.2853f, -16.4931f), 0.2147f, vec3(0.562f, 0.940f, 0.676f),
     false},
    {vec3(0.4240f, -0.1980f, -3.4605f), 0.3020f, vec3(0.892f, 0.500f, 0.709f),
     false},
    {vec3(2.8370f, -0.3257f, -16.5435f), 0.1743f, vec3(0.341f, 0.277f, 0.591f),
     false},
    {vec3(-5.5312f, -0.1674f, -7.8446f), 0.3326f, vec3(0.816f, 0.549f, 0.723f),
     false},
    {vec3(0.2052f, -0.2733f, -15.6718f), 0.2267f, vec3(0.853f, 0.841f, 0.371f),
     false},
    {vec3(5.8213f, -0.1920f, -7.0540f), 0.3080f, vec3(0.556f, 0.806f, 0.376f),
     false},
    {vec3(0.2843f, -0.2521f, -2.5261f), 0.2479f, vec3(0.426f, 0.592f, 0.585f),
     false},
    {vec3(-3.5162f, -0.2589f, -5.4702f), 0.2411f, vec3(0.708f, 0.607f, 0.337f),
     false},
    {vec3(-2.5509f, -0.2090f, -13.7940f), 0.2910f, vec3(0.681f, 0.807f, 0.935f),
     false},
    {vec3(8.8733f, -0.2320f, -14.7849f), 0.2680f, vec3(0.502f, 0.571f, 0.516f),
     false},
    {vec3(6.8700f, -0.3191f, -7.9813f), 0.1809f, vec3(0.421f, 0.704f, 0.670f),
     false},
    {vec3(-11.0335f, -0.1981f, -14.8305f), 0.3019f,
     vec3(0.205f, 0.655f, 0.887f), false},
    {vec3(7.9598f, -0.2613f, -11.7980f), 0.2387f, vec3(0.187f, 0.787f, 0.385f),
     false},
    {vec3(-7.2484f, -0.2368f, -13.4728f), 0.2632f, vec3(0.213f, 0.847f, 0.646f),
     false},
    {vec3(-8.8117f, -0.3018f, -15.5181f), 0.1982f, vec3(0.519f, 0.353f, 0.354f),
     false},
    {vec3(8.9222f, -0.3481f, -13.6788f), 0.1519f, vec3(0.692f, 0.276f, 0.503f),
     false},
    {vec3(-0.2136f, -0.3102f, -6.5398f), 0.1898f, vec3(0.340f, 0.608f, 0.610f),
     false},
    {vec3(0.6739f, -0.2184f, -4.6662f), 0.2816f, vec3(0.699f, 0.746f, 0.189f),
     false},
    {vec3(6.1671f, -0.2287f, -8.4444f), 0.2713f, vec3(0.379f, 0.789f, 0.636f),
     false},
    {vec3(2.2280f, -0.2795f, -10.8225f), 0.2205f, vec3(0.692f, 0.727f, 0.677f),
     false},
    {vec3(7.3582f, -0.1823f, -10.6802f), 0.3177f, vec3(0.667f, 0.397f, 0.503f),
     false},
    {vec3(5.2683f, -0.1567f, -14.6224f), 0.3433f, vec3(0.934f, 0.924f, 0.794f),
     false},
    {vec3(-10.6559f, -0.2768f, -13.4416f), 0.2232f,
     vec3(0.579f, 0.514f, 0.688f), false},
    {vec3(5.5617f, -0.2155f, -9.9375f), 0.2845f, vec3(0.902f, 0.237f, 0.337f),
     false},
    {vec3(1.4764f, -0.3450f, -15.0320f), 0.1550f, vec3(0.882f, 0.327f, 0.201f),
     false},
    {vec3(-11.9474f, -0.2684f, -16.1759f), 0.2316f,
     vec3(0.446f, 0.505f, 0.910f), false},
    {vec3(2.3287f, -0.1804f, -7.7302f), 0.3196f, vec3(0.663f, 0.628f, 0.167f),
     false},
    {vec3(-3.5615f, -0.1926f, -4.1407f), 0.3074f, vec3(0.602f, 0.205f, 0.762f),
     false},
    {vec3(3.2884f, -0.3086f, -3.6712f), 0.1914f, vec3(0.413f, 0.268f, 0.870f),
     false},
    {vec3(1.2540f, -0.3152f, -0.8908f), 0.1848f, vec3(0.571f, 0.747f, 0.531f),
     false},
    {vec3(0.5074f, -0.1934f, -14.7387f), 0.3066f, vec3(0.516f, 0.921f, 0.199f),
     false},
    {vec3(-7.7257f, -0.3338f, -10.3411f), 0.1662f, vec3(0.370f, 0.656f, 0.589f),
     false},
    {vec3(0.8111f, -0.2850f, -16.9087f), 0.2150f, vec3(0.513f, 0.634f, 0.229f),
     false},
    {vec3(3.5209f, -0.2096f, -14.4975f), 0.2904f, vec3(0.765f, 0.727f, 0.322f),
     false},
    {vec3(-5.4376f, -0.2593f, -7.0718f), 0.2407f, vec3(0.491f, 0.682f, 0.449f),
     false},
    {vec3(1.2627f, -0.2022f, -13.8001f), 0.2978f, vec3(0.619f, 0.599f, 0.414f),
     false},
    {vec3(1.9865f, -0.3256f, -6.0112f), 0.1744f, vec3(0.750f, 0.844f, 0.727f),
     false},
    {vec3(-2.6124f, -0.1563f, -10.2070f), 0.3437f, vec3(0.612f, 0.320f, 0.675f),
     false},
    {vec3(-1.6705f, -0.1760f, -9.6275f), 0.3240f, vec3(0.263f, 0.657f, 0.175f),
     false},
    {vec3(-0.7116f, -0.2008f, -3.6573f), 0.2992f, vec3(0.423f, 0.446f, 0.727f),
     false},
    {vec3(0.8918f, -0.3395f, -2.6760f), 0.1605f, vec3(0.689f, 0.368f, 0.680f),
     false},
    {vec3(-0.2590f, -0.2934f, -11.5343f), 0.2066f, vec3(0.684f, 0.186f, 0.466f),
     false},
    {vec3(7.0087f, -0.2499f, -14.1670f), 0.2501f, vec3(0.208f, 0.839f, 0.184f),
     false},
    {vec3(2.6180f, -0.2850f, -13.6225f), 0.2150f, vec3(0.816f, 0.886f, 0.221f),
     false},
    {vec3(-5.9271f, -0.1862f, -13.8556f), 0.3138f, vec3(0.267f, 0.308f, 0.632f),
     false},
    {vec3(-6.0884f, -0.1980f, -11.1437f), 0.3020f, vec3(0.768f, 0.545f, 0.754f),
     false},
    {vec3(0.0387f, -0.2951f, -0.9549f), 0.2049f, vec3(0.398f, 0.512f, 0.196f),
     false},
    {vec3(0.1268f, -0.1789f, -10.4551f), 0.3211f, vec3(0.520f, 0.593f, 0.783f),
     false},
    {vec3(-0.2751f, -0.2196f, -5.4660f), 0.2804f, vec3(0.271f, 0.199f, 0.233f),
     false},
    {vec3(-2.7033f, -0.3183f, -6.3385f), 0.1817f, vec3(0.477f, 0.421f, 0.628f),
     false},
    {vec3(-8.1054f, -0.1922f, -11.0042f), 0.3078f, vec3(0.226f, 0.693f, 0.377f),
     false},
    {vec3(5.5793f, -0.3217f, -5.9470f), 0.1783f, vec3(0.709f, 0.464f, 0.626f),
     false},
    {vec3(3.8465f, -0.1842f, -12.5592f), 0.3158f, vec3(0.571f, 0.667f, 0.489f),
     false},
    {vec3(-3.9068f, -0.2776f, -6.1642f), 0.2224f, vec3(0.321f, 0.908f, 0.539f),
     false},
    {vec3(1.7045f, -0.1811f, -1.7194f), 0.3189f, vec3(0.818f, 0.857f, 0.180f),
     false},
    {vec3(-5.6440f, -0.2598f, -5.9429f), 0.2402f, vec3(0.193f, 0.552f, 0.339f),
     false},
    {vec3(3.9724f, -0.3405f, -7.1370f), 0.1595f, vec3(0.516f, 0.239f, 0.874f),
     false},
    {vec3(-0.5133f, -0.3016f, -2.4408f), 0.1984f, vec3(0.642f, 0.342f, 0.483f),
     false},
    {vec3(0.1167f, -0.3365f, -8.9430f), 0.1635f, vec3(0.941f, 0.593f, 0.462f),
     false},
    {vec3(8.8552f, -0.2560f, -10.8064f), 0.2440f, vec3(0.353f, 0.163f, 0.781f),
     false},
    {vec3(5.7064f, -0.2109f, -10.7395f), 0.2891f, vec3(0.200f, 0.771f, 0.516f),
     false},
    {vec3(4.8070f, -0.3251f, -6.2563f), 0.1749f, vec3(0.745f, 0.866f, 0.459f),
     false},
    {vec3(-2.9257f, -0.1855f, -9.2731f), 0.3145f, vec3(0.214f, 0.679f, 0.395f),
     false},
    {vec3(1.8354f, -0.2797f, -0.7200f), 0.2203f, vec3(0.432f, 0.949f, 0.370f),
     false},
    {vec3(-2.8935f, -0.3366f, -13.2348f), 0.1634f, vec3(0.456f, 0.604f, 0.634f),
     false},
    {vec3(-3.2645f, -0.2142f, -16.1300f), 0.2858f, vec3(0.760f, 0.609f, 0.574f),
     false},
    {vec3(-1.1189f, -0.3255f, -14.4980f), 0.1745f, vec3(0.869f, 0.506f, 0.220f),
     false},
    {vec3(5.8345f, -0.2881f, -16.9931f), 0.2119f, vec3(0.571f, 0.765f, 0.809f),
     false},
    {vec3(3.7066f, -0.3352f, -16.5305f), 0.1648f, vec3(0.510f, 0.694f, 0.426f),
     false},
    {vec3(-6.1793f, -0.2230f, -9.3643f), 0.2770f, vec3(0.392f, 0.577f, 0.552f),
     false},
    {vec3(5.0079f, -0.2346f, -11.0692f), 0.2654f, vec3(0.483f, 0.661f, 0.548f),
     false},
    {vec3(4.8249f, -0.2246f, -4.9244f), 0.2754f, vec3(0.536f, 0.794f, 0.698f),
     false},
    {vec3(-4.4327f, -0.2621f, -8.2323f), 0.2379f, vec3(0.635f, 0.400f, 0.725f),
     false},
    {vec3(-3.3887f, -0.2862f, -7.9300f), 0.2138f, vec3(0.348f, 0.227f, 0.382f),
     false},
    {vec3(5.4065f, -0.1735f, -12.3676f), 0.3265f, vec3(0.443f, 0.470f, 0.606f),
     false},
    {vec3(-4.6978f, -0.2491f, -12.9949f), 0.2509f, vec3(0.941f, 0.694f, 0.245f),
     false},
    {vec3(6.4462f, -0.2822f, -15.9621f), 0.2178f, vec3(0.309f, 0.557f, 0.550f),
     false},
    {vec3(-1.1864f, -0.3409f, -2.3296f), 0.1591f, vec3(0.529f, 0.516f, 0.635f),
     false},
    {vec3(-4.0172f, -0.2435f, -12.0486f), 0.2565f, vec3(0.803f, 0.445f, 0.689f),
     false},
    {vec3(-9.2757f, -0.2049f, -11.6967f), 0.2951f, vec3(0.530f, 0.924f, 0.776f),
     false},
    {vec3(-1.7612f, -0.3197f, -11.7385f), 0.1803f, vec3(0.770f, 0.886f, 0.848f),
     false},
    {vec3(8.3469f, -0.2741f, -16.6627f), 0.2259f, vec3(0.914f, 0.794f, 0.382f),
     false},
    {vec3(-0.2053f, -0.2615f, -4.3595f), 0.2385f, vec3(0.312f, 0.581f, 0.896f),
     false},
    {vec3(3.0458f, -0.1674f, -2.9761f), 0.3326f, vec3(0.291f, 0.886f, 0.948f),
     false},
    {vec3(1.9451f, -0.2985f, -3.1654f), 0.2015f, vec3(0.783f, 0.604f, 0.756f),
     false},
    {vec3(-9.9595f, -0.2896f, -12.2890f), 0.2104f, vec3(0.673f, 0.704f, 0.200f),
     false},
    {vec3(-5.0364f, -0.2910f, -10.4486f), 0.2090f, vec3(0.257f, 0.273f, 0.748f),
     false},
    {vec3(10.2166f, -0.1962f, -13.0191f), 0.3038f, vec3(0.667f, 0.486f, 0.944f),
     false},
    {vec3(9.6450f, -0.2735f, -14.7835f), 0.2265f, vec3(0.451f, 0.696f, 0.679f),
     false},
    {vec3(5.2071f, -0.2921f, -13.8684f), 0.2079f, vec3(0.405f, 0.628f, 0.688f),
     false},
    {vec3(-1.3242f, -0.1677f, -10.6419f), 0.3323f, vec3(0.546f, 0.928f, 0.903f),
     false},
    {vec3(-3.9537f, -0.2157f, -13.3587f), 0.2843f, vec3(0.483f, 0.269f, 0.451f),
     false},
    {vec3(6.1273f, -0.2899f, -12.0288f), 0.2101f, vec3(0.882f, 0.600f, 0.924f),
     false},
    {vec3(-3.9601f, -0.2541f, -14.7811f), 0.2459f, vec3(0.916f, 0.162f, 0.900f),
     false},
    {vec3(1.2886f, -0.3091f, -15.5571f), 0.1909f, vec3(0.770f, 0.455f, 0.577f),
     false},
    {vec3(-1.8682f, -0.2164f, -12.9938f), 0.2836f, vec3(0.824f, 0.335f, 0.716f),
     false},
    {vec3(-12.6297f, -0.2951f, -16.6171f), 0.2049f,
     vec3(0.801f, 0.689f, 0.795f), false},
    {vec3(-4.6475f, -0.3434f, -15.7890f), 0.1566f, vec3(0.919f, 0.620f, 0.752f),
     false},
    {vec3(-0.2490f, -0.2953f, -16.8740f), 0.2047f, vec3(0.435f, 0.903f, 0.495f),
     false},
    {vec3(1.5867f, -0.3336f, -2.6218f), 0.1664f, vec3(0.657f, 0.742f, 0.403f),
     false},
    {vec3(9.8975f, -0.3331f, -15.3193f), 0.1669f, vec3(0.933f, 0.608f, 0.286f),
     false},
    {vec3(8.0832f, -0.2831f, -13.8631f), 0.2169f, vec3(0.929f, 0.259f, 0.407f),
     false},
    {vec3(3.9638f, -0.3015f, -6.3929f), 0.1985f, vec3(0.464f, 0.241f, 0.601f),
     false},
};

void gen_scene(int W, int H) {
  sphere *dScene;
  cudaMalloc(&dScene, N * sizeof(sphere));
  cudaMemcpy(dScene, hScene, N * sizeof(sphere), cudaMemcpyHostToDevice);

  int *hPpm, *dPpm;
  hPpm = (int *)malloc(3 * H * W * sizeof(int));
  cudaMalloc(&dPpm, 3 * H * W * sizeof(int));

  const int TX = 16;
  const int TY = 16;
  dim3 block(TX, TY, 1);
  dim3 grid((W + TX - 1) / TX, (H + TY - 1) / TY, 1);

#ifdef PERF
  // Warm up runs (maybe too many iters)
  for (int i = 0; i < 10; i++) {
    gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  }
  cudaDeviceSynchronize();

  const unsigned iters = 100;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < iters; i++) {
    gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Execution Time (Avg over %d iterations): %f ms\n", iters,
         milliseconds / iters);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif

  gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  cudaMemcpy(hPpm, dPpm, 3 * H * W * sizeof(int),
             cudaMemcpyDeviceToHost); // this takes care of sync, no need of CDS

  unsigned char *hPpmChar =
      (unsigned char *)malloc(3 * H * W * sizeof(unsigned char));

  char filePath[255];
  snprintf(filePath, sizeof(filePath), "%s/gpu_image.ppm",
           std::getenv("OUTDIR"));
  FILE *f = fopen(filePath, "wb");
  fprintf(f, "P6\n%d %d\n255\n", W, H);
  for (int i = 0; i < (W * H); i++) {
    int pixel_idx = 3 * i;
    hPpmChar[pixel_idx] = (unsigned char)hPpm[pixel_idx];
    hPpmChar[pixel_idx + 1] = (unsigned char)hPpm[pixel_idx + 1];
    hPpmChar[pixel_idx + 2] = (unsigned char)hPpm[pixel_idx + 2];
  }
  fwrite(hPpmChar, 1, 3 * W * H * sizeof(unsigned char), f);
  fclose(f);
  printf("Wrote gpu_image.ppm (%dx%d)\n", W, H);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <width> <height>" << endl;
    std::cout << "Example: " << argv[0] << " 1920 1080" << endl;
    return 1;
  }
  const int W = atoi(argv[1]), H = atoi(argv[2]);
  gen_scene(W, H);
  return 0;
}

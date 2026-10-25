#include "sphere.cuh"
#include "vec3.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cuda.h>
#include <iostream>

using namespace std;
namespace cg = cooperative_groups;

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
  float k = 0.2;
  float tHit;
  int hit = closestHit(origin, dir, tHit, scene, 4);

  if (hit < 0) {
    // Miss: draw a simple sky gradient (white near horizon -> blue up top).
    float t = 0.5f * (dir.y() + 1.0f);
    return vec3(1, 1, 1) * (1 - t) + vec3(0.4f, 0.6f, 1.0f) * t;
  }

  // The point we hit, and the surface normal there.
  vec3 p = origin + dir * tHit;
  vec3 n = normalize(p - scene[hit].center);

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
  int blocked = closestHit(p + n * 0.001f, toLight, shadowT, scene, 4);
  float shade = (blocked >= 0) ? 0.3f : 1.0f;

  float brightness = 0.15f + diffuse * shade; // 0.15 = a little ambient light
                                              //
  vec3 rDir = normalize(dir - (2 * dot(dir, n) * n));
  if (bounce == 0) {
    return base * brightness;
  } else {
    return (1 - k) * base * brightness +
           k * rayColor(p + rDir * 0.01f, rDir, scene, bounce - 1);
  }
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

    // vec3 color = rayColor(cameraPos, dir, scene[0]);
    vec3 color = rayColor(cameraPos, dir, scene, 1);

    auto r = (int)(fminf(color.x(), 1.0f) * 255);
    auto g = (int)(fminf(color.y(), 1.0f) * 255);
    auto b = (int)(fminf(color.z(), 1.0f) * 255);

    auto pixel = vec3(r, g, b);
    out[outIdx] = pixel.x();
    out[outIdx + 1] = pixel.y();
    out[outIdx + 2] = pixel.z();
  }
}

sphere hScene[4] = {
    {vec3(0, -1000.5f, -1), 1000.0f, vec3(0.5f, 0.5f, 0.5f), true}, //
    {vec3(0, 0.0f, -1), 0.5f, vec3(0.9f, 0.2f, 0.2f), false},
    {vec3(-1.1f, 0.0f, -1), 0.5f, vec3(0.2f, 0.7f, 0.3f), false},
    {vec3(1.1f, 0.0f, -1), 0.5f, vec3(0.2f, 0.4f, 0.9f), false},
};

void gen_scene(int W, int H) {
  sphere *dScene;
  cudaMalloc(&dScene, 4 * sizeof(sphere));
  cudaMemcpy(dScene, hScene, 4 * sizeof(sphere), cudaMemcpyHostToDevice);

  int *hPpm, *dPpm;
  hPpm = (int *)malloc(3 * H * W * sizeof(int));
  cudaMalloc(&dPpm, 3 * H * W * sizeof(int));

  const int TX = 16;
  const int TY = 16;
  dim3 block(TX, TY, 1);
  dim3 grid((W + TX - 1) / TX, (H + TY - 1) / TY, 1);

#ifdef DEBUG
  // Warm up runs (maybe too many iters)
  for (int i = 0; i < 10; i++) {
    gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  }
  cudaDeviceSynchronize();

  const unsigned N = 100;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < N; i++) {
    gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel Execution Time (Avg over %d iterations): %f ms\n", N,
         milliseconds / N);
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
  // const int W = 256;
  // const int H = 256;
  //
  // int *hPpm, *dPpm;
  // hPpm = (int *)malloc(3 * H * W * sizeof(int));
  // cudaMalloc(&dPpm, 3 * H * W * sizeof(int));
  //
  // // TODO: Tic-toc it
  // gen_ppm(W, H, dPpm, hPpm);
  //
  // cudaFree(dPpm);
  // free(hPpm);

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <width> <height>" << endl;
    std::cout << "Example: " << argv[0] << " 1920 1080" << endl;
    return 1;
  }
  const int W = atoi(argv[1]), H = atoi(argv[2]);
  gen_scene(W, H);
  return 0;
}

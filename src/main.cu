#include "sphere.cuh"
#include "vec3.cuh"
#include <cmath>
#include <cuda.h>
#include <iostream>

using namespace std;

__global__ void cuda_gen_gradient(int W, int H, int *out) {
  const float ALPHA = 255.999; // scaling factor

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int outIdx = 3 * ((y * W) + x);
  if (y < H && x < W) {
    auto r = double(x) / (W - 1);
    auto g = double(y) / (H - 1);
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
  const int TX = 16;
  const int TY = 16;
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
    float u = (2.0f * (px + 0.5f) / W - 1.0f) * (double)(W / H);
    float v = (1.0f - 2.0f * (py + 0.5f) / H);

    vec3 cameraPos = vec3(0, 1, 3);
    vec3 dir = normalize(vec3(u, v, -1.5f));

    // vec3 color = rayColor(cameraPos, dir, scene[0]);
    vec3 color = rayColor(cameraPos, dir, scene);

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
  gen_scene_kernel<<<grid, block>>>(W, H, dScene, dPpm);
  cudaDeviceSynchronize(); // redundant but good practice
  cudaMemcpy(hPpm, dPpm, 3 * H * W * sizeof(int), cudaMemcpyDeviceToHost);

  unsigned char *hPpmChar =
      (unsigned char *)malloc(3 * H * W * sizeof(unsigned char));

  FILE *f = fopen("../results/gpu_image.ppm", "wb");
  fprintf(f, "P6\n%d %d\n255\n", W, H);
  for (int i = 0; i < (W * H); i++) {
    int base_idx = 3 * i;
    hPpmChar[base_idx] = (unsigned char)hPpm[base_idx];
    hPpmChar[base_idx + 1] = (unsigned char)hPpm[base_idx + 1];
    hPpmChar[base_idx + 2] = (unsigned char)hPpm[base_idx + 2];
  }
  fwrite(hPpmChar, 1, 3 * W * H * sizeof(unsigned char), f);
  fclose(f);
  printf("Wrote image.ppm (%dx%d)\n", W, H);
}

int main() {
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

  const int W = 3840, H = 2160;
  gen_scene(W, H);
  return 0;
}

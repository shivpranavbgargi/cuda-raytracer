#include "vec3.cuh"
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

int main() {
  const int W = 256;
  const int H = 256;

  int *hPpm, *dPpm;
  hPpm = (int *)malloc(3 * H * W * sizeof(int));
  cudaMalloc(&dPpm, 3 * H * W * sizeof(int));

  // TODO: Tic-toc it
  gen_ppm(W, H, dPpm, hPpm);

  cudaFree(dPpm);
  free(hPpm);
}

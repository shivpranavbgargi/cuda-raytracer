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
    out[outIdx] = int(ALPHA * r);
    out[outIdx + 1] = int(ALPHA * g);
    out[outIdx + 2] = int(ALPHA * b);
  }
}

void gen_ppm(int W, int H, int *dPpm, int *hPpm) {
  dim3 block(16, 16, 1);
  dim3 grid((W + 15) / 16, (H + 15) / 16, 1);

  cuda_gen_gradient<<<grid, block>>>(W, H, dPpm);
  cudaDeviceSynchronize();
  cudaMemcpy(hPpm, dPpm, 3 * H * W * sizeof(int), cudaMemcpyDeviceToHost);

  cout << "P3\n" << W << ' ' << H << "\n255\n";
  for (int i = 0; i < (W * H); i++) {
    int base_idx = 3 * i;
    cout << hPpm[base_idx] << ' ' << hPpm[base_idx + 1] << ' '
         << hPpm[base_idx + 2] << '\n';
  }
}

int main() {
  int W = 256;
  int H = 256;

  int *hPpm, *dPpm;
  hPpm = (int *)malloc(3 * H * W * sizeof(int));
  cudaMalloc(&dPpm, 3 * H * W * sizeof(int));

  gen_ppm(W, H, dPpm, hPpm);

  cudaFree(dPpm);
  free(hPpm);
}

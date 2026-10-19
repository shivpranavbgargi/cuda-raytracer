#ifndef CUDARAYTRACER_CUH
#define SPHERE_CUH
void gen_scene(int W, int H);
__global__ void gen_scene_kernel(int W, int H, sphere *scene, int *out);
#endif

# CUDA Ray Tracer

- [Abstract](#abstract)
- [Hardware and Software](#hardware-and-software)
- [Raytraced image](#ray-traced-image)
- [References](#references)

## Abstract

Implementation of a ray tracer using NVIDIA CUDA.

---

## Ray-traced image

### V1 (Shadows)

[v1_gpu_image.ppm](docs/images/v1_gpu_image.ppm)

![v1 .jpg image](docs/images/v1_gpu_image.jpg)

### V2 (Reflection - Single Bounce)

[v2_gpu_image.ppm](docs/images/v2_gpu_image.ppm)

![v2 .jpg image](docs/images/v2_gpu_image.jpg)

![v2 .jpg image_zoomed](docs/images/v2_gpu_image_zoomed.jpg)

### V3 (Reflection - Multiple Bounces)

[v3_gpu_image.ppm](docs/images/v3_gpu_image.ppm)

![v3 .jpg image](docs/images/v3_gpu_image.jpg)

![v3 .jpg image_zoomed](docs/images/v3_gpu_image_zoomed.jpg)

---

## Hardware and Software

- **OS:** Arch Linux
- **Kernel:** 7.2.8-arch1-2
- **CPU:** Intel Core i7-13700K (13th Gen)
- **CPU cores:** 24 logical CPUs (16 physical cores)
- **RAM:** 62 GiB
- **GPU:** NVIDIA GeForce RTX 3080
- **GPU VRAM:** 10 GiB
- **NVIDIA Driver:** 615.71.09
- **CUDA Toolkit:** 13.4
- **GCC:** 16.2.1

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

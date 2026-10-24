#include <cstdio>
#include <iostream>
#include <string.h>
#include <vector>

int tests_passed = 0;
int tests_failed = 0;

std::vector<unsigned char> read_ppm_rgb(const char *filePath, int *W, int *H) {
  char buf[1024];
  char tmp[1024];
  FILE *f = fopen(filePath, "rb");
  if (f == NULL) {
    std::cout << "Unable to open the file: " << filePath << std::endl;
    std::vector<unsigned char> empty;
    return empty;
  }
  for (int i = 0; i < 2; i++) {
    fgets(buf, sizeof(buf), f);
  }
  strcpy(tmp, buf);
  char *space = strchr(buf, ' ');
  space++;
  *W = atoi(buf);
  *H = atoi(space);

  long start = ftell(f);
  fseek(f, 0, SEEK_END);
  size_t n = (ftell(f) - start) / sizeof(unsigned char);
  fseek(f, start, SEEK_SET);

  std::vector<unsigned char> rgb(n);
  fread(rgb.data(), sizeof(unsigned char), n, f);

  fclose(f);
  return rgb;
}

bool chk_err(int w, int h) {
#ifdef VERBOSE
  std::cout << "==================================" << std::endl;
  std::cout << "Checking for Floating point errors" << std::endl;
  std::cout << "==================================" << std::endl;
#endif

  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "%s/main_gpu %d %d 1> /dev/null", std::getenv("BUILD_DIR"), w,
           h);
  int result = std::system(cmd);
  snprintf(cmd, sizeof(cmd), "%s/main_cpu %d %d 1> /dev/null", std::getenv("BUILD_DIR"), w,
           h);
  result = std::system(cmd);

  char filePath[255];
  int W, H;
  snprintf(filePath, sizeof(filePath), "%s/gpu_image.ppm",
           std::getenv("OUTDIR"));
  std::vector<unsigned char> gpu_rgb = read_ppm_rgb(filePath, &W, &H);
  snprintf(filePath, sizeof(filePath), "%s/cpu_image.ppm",
           std::getenv("OUTDIR"));
  std::vector<unsigned char> cpu_rgb = read_ppm_rgb(filePath, &W, &H);

  bool ret = true;
  unsigned int count = 0;
  unsigned int critical_count = 0;
  float abs_err;
  for (std::size_t i = 0; i < gpu_rgb.size(); i++) {
    int gpu_subpix = (int)gpu_rgb[i];
    int cpu_subpix = (int)cpu_rgb[i];
    if (gpu_subpix != cpu_subpix) {
      abs_err = (float)std::abs(gpu_subpix - cpu_subpix) / 255.0f;
      unsigned px = (i / 3) % W;
      unsigned py = (i / 3) / W;
#ifdef VERBOSE
      std::cout << i << ": (" << px << ", " << py << ")" << ", "
                << std::abs(gpu_subpix - cpu_subpix) << ", " << abs_err;
#endif
      if (abs_err > 1e-2) {
#ifdef VERBOSE
        std::cout << ", " << "CRITICAL";
#endif
        critical_count++;
        ret = false;
      }
#ifdef VERBOSE
      std::cout << std::endl;
#endif
      count++;
    }
  }
#ifdef VERBOSE
  std::cout << "Number of subpixels in error: " << count << std::endl;
  std::cout << "Number of subpixels in error (critical): " << critical_count
            << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "==================================" << std::endl;
#endif
  return ret;
}

void test(bool condition, const char *test_name) {
  if (condition) {
    std::cout << "[PASS] " << test_name << '\n';
    tests_passed++;
  } else {
    std::cout << "[FAIL] " << test_name << '\n';
    tests_failed++;
  }
}

const int res[5][2] = {
    {1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}, {7680, 4320}};
int main() {
  std::cout<< "=======" << std::endl;
  std::cout<< "TESTING" << std::endl;
  std::cout<< "=======" << std::endl;
  for (int i = 0; i < 5; i++) {
    char msg[1024];
    snprintf(msg, sizeof(msg), "Floating point errors (%dx%d)", res[i][0],
             res[i][1]);
    test(chk_err(res[i][0], res[i][1]) == true, msg);
  }

  std::cout << "\nPassed: " << tests_passed << ", Failed: " << tests_failed
            << '\n';

  return tests_failed == 0 ? 0 : 1;
}

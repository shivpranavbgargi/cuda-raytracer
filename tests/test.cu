#include <iostream>

int add(int a, int b) { return a + b; }

int tests_passed = 0;
int tests_failed = 0;

void test(bool condition, const char *test_name) {
  if (condition) {
    std::cout << "[PASS] " << test_name << '\n';
    tests_passed++;
  } else {
    std::cout << "[FAIL] " << test_name << '\n';
    tests_failed++;
  }
}

int main() {
  test(add(2, 3) == 5, "add positive numbers");
  test(add(0, 0) == 0, "add zeros");
  test(add(-2, 3) == 1, "add negative and positive");
  test(add(-2, -3) == -5, "add negative numbers");

  std::cout << "\nPassed: " << tests_passed << ", Failed: " << tests_failed
            << '\n';

  return tests_failed == 0 ? 0 : 1;
}

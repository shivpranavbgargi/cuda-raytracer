NVCC = nvcc
GPP = g++
BUILD_DIR = bin
TEST_DIR := tests
SRC_DIR := src

CPU_CFLAGS = -O3 -ffp-contract=off
TEST_CFLAGS = -O3
GPU_CFLAGS = -O3 --fmad=false
DEBUG_FLAGS = -O3 -lineinfo

SRC = $(ROOT)/$(SRC_DIR)/main.cu
SRC_C = $(ROOT)/$(SRC_DIR)/main.cpp
SRC_TEST = $(ROOT)/$(TEST_DIR)/tests.cpp

TARGET = $(BUILD_DIR)/$(GPU_BIN)
DEBUG_TARGET = $(BUILD_DIR)/$(DEBUG_GPU_BIN)
CPU_TARGET = $(BUILD_DIR)/$(CPU_BIN)
TEST_TARGET = $(BUILD_DIR)/tests

.PHONY: all clean

all: $(TARGET) $(CPU_TARGET)
debug: $(DEBUG_TARGET)
test: $(TARGET) $(CPU_TARGET) $(TEST_TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(GPU_CFLAGS) $(SRC) -o $@ -I$(ROOT)/include/

$(DEBUG_TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(DEBUG_FLAGS) $(SRC) -o $@ -I$(ROOT)/include/ -DDEBUG

$(CPU_TARGET): $(SRC_C) | $(BUILD_DIR)
	$(GPP) $(CPU_CFLAGS) $(SRC_C) -o $@ -I$(ROOT)/include/

$(TEST_TARGET): $(SRC_TEST) | $(BUILD_DIR)
	$(GPP) $(TEST_CFLAGS) $(SRC_TEST) -o $@ -I$(ROOT)/include/
	./$(BUILD_DIR)/tests

clean:
	rm -rf $(BUILD_DIR)

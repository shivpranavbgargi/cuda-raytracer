NVCC = nvcc
GPP = g++
OUTDIR = bin
CPU_CFLAGS = -O3
GPU_CFLAGS = -O3 -lineinfo
SRC = src/main.cu
SRC_C = src/main.cpp
TARGET = $(OUTDIR)/$(GPU_BIN)
DEBUG_TARGET = $(OUTDIR)/$(DEBUG_GPU_BIN)
CPU_TARGET = $(OUTDIR)/$(CPU_BIN)

.PHONY: all clean

all: $(TARGET) $(DEBUG_TARGET) $(CPU_TARGET)
debug: $(DEBUG_TARGET)

$(OUTDIR):
	mkdir -p $@

$(TARGET): $(SRC) | $(OUTDIR)
	$(NVCC) $(GPU_CFLAGS) $(SRC) -o $@ -I./include/

$(DEBUG_TARGET): $(SRC) | $(OUTDIR)
	$(NVCC) $(GPU_CFLAGS) $(SRC) -o $@ -I./include/ -DDEBUG

$(CPU_TARGET): $(SRC_C) | $(OUTDIR)
	$(GPP) $(CPU_CFLAGS) $(SRC_C) -o $@ -I./include/

clean:
	rm -rf $(OUTDIR) $(DEBUG_TARGET)

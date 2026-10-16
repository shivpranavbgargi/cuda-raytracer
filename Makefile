NVCC = nvcc
OUTDIR = bin
CFLAGS = -O3
SRC = src/main.cu
TARGET = $(OUTDIR)/main

.PHONY: all clean

all: $(TARGET)

$(OUTDIR):
	mkdir -p $@

$(TARGET): $(SRC) | $(OUTDIR)
	$(NVCC) $(CFLAGS) $(SRC) -o $@ -I./include/

clean:
	rm -rf $(OUTDIR)


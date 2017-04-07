mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

CUDA_HOME = /usr/local/cuda
CUDA_LIB = ${CUDA_HOME}/lib

NVCC = ${CUDA_HOME}/bin/nvcc

DEFINES=
CFLAGS = ${DEFINES} -I ${current_dir}/src

LIBS=-std=c++11

CUDA_ARCH=-arch=sm_35
CUDA_LDFLAGS=-L${CUDA_HOME}/lib64 -lcudart 

SOURCES=src/main.cu
TARGET=/build/ece508-convlayer
DATA=data/0

ARGS = 

.DEFAULT: $(TARGET)
.PHONY: memcheck

$(TARGET): $(SOURCES)
	$(NVCC) ${CFLAGS} $< -o $@ -O3 ${LIBS} ${CUDA_ARCH}


memcheck: $(TARGET)
	$(CUDA_HOME)/bin/cuda-memcheck $< $(ARGS)

clean:
	rm -rf *.o $(TARGET)

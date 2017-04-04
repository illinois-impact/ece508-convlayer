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
TARGET=ece508-convlayer
DATA=data/0

ARGS = 

.DEFAULT: cnn
.PHONY: run memcheck

$(TARGET): $(SOURCES)
	$(NVCC) ${CFLAGS} $< -o $@ -g -O0 ${LIBS} ${CUDA_ARCH}

run: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) ./$(TARGET) $(ARGS)

memcheck: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) $(CUDA_HOME)/bin/cuda-memcheck ./$(TARGET) $(ARGS)

clean:
	rm -rf *.o $(TARGET)

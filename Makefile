CUDA_HOME = /usr/local/cuda
CUDA_LIB = $(CUDA_HOME)/lib

NVCC = nvcc

DEFINES=
CFLAGS = ${DEFINES}

LIBS=-lrt -lcr -lpthread -lm
LIBS+=-std=c++11

CUDA_ARCH=-
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart 

TARGET = ece508-convlayer
DATA = data/0

ARGS = 

.DEFAULT: cnn
.PHONY: run memcheck

cnn: template.cu
	$(NVCC) ${CFLAGS} main.cu -o $@ -g -O0 -lwb -Xlinker='-Bsymbolic-functions -z relro' ${LIBS}
	#$(NVCC) ${CFLAGS} template.cu -o $@ -g -O0 -I$(LIBWB_INC) -L$(LIBWB_LIB) -lwb -Xlinker='-Bsymbolic-functions -z relro' ${LIBS}

run: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) ./$(TARGET) $(ARGS)

memcheck: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) $(CUDA_HOME)/bin/cuda-memcheck ./$(TARGET) $(ARGS)

clean:
	rm -rf *.o $(TARGET)

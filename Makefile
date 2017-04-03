CUDA_HOME = /usr/local/cuda
CUDA_LIB = $(CUDA_HOME)/lib
LIBWB_INC = ${HOME}/wbgo/src/wb/c-tools
LIBWB_LIB = ${LIBWB_INC}/Linux-x86_64

#NVCC = /usr/local/cuda-8.0/vin/nvcc
NVCC = nvcc

DEFINES=-DWB_USE_CUDA -DWB_USE_MPI 
DEFINES+=-DWB_USE_COURSERA -DWB_USE_CUSTOM_MALLOC 
#DEFINES+=-w -arch -sm -l
CFLAGS = ${DEFINES}
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart 
LIBS=-lrt -lcr -lpthread -lm
LIBS+=-std=c++11

DATA = data/0

ARGS = -i $(DATA)/mode.flag,$(DATA)/input_desc.raw -e $(DATA)/expected_out_md5.raw -t integral_vector

.DEFAULT: cnn
.PHONY: run memcheck debug

cnn: template.cu
	$(NVCC) ${CFLAGS} main.cu -o $@ -g -O0 -lwb -Xlinker='-Bsymbolic-functions -z relro' ${LIBS}
	#$(NVCC) ${CFLAGS} template.cu -o $@ -g -O0 -I$(LIBWB_INC) -L$(LIBWB_LIB) -lwb -Xlinker='-Bsymbolic-functions -z relro' ${LIBS}

run: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) mpirun -np 5 ./cnn $(ARGS)

memcheck: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) mpirun -np 5 $(CUDA_HOME)/bin/cuda-memcheck ./cnn $(ARGS)

debug: cnn
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(CUDA_LIB):$(LIBWB_LIB) mpirun -np 5 xterm -e gdb -ex 'show environment MV2_COMM_WORLD_RANK' -ex r --args ./cnn $(ARGS)

clean:
	rm -rf *.o cnn

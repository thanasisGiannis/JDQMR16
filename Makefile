SRC=./src/
HFILES = $(SRC)*.h
CUDADIR=/usr/local/cuda-10.2

CUDA_ARCH_FLAGS := -O3 -w -arch=sm_70 -g -G 
CFLAGS = $(CUDA_ARCH_FLAGS)

CC = $(CUDADIR)/bin/nvcc
INCLUDES= -I$(CUDADIR)/include 
LIBS =  -lm -lcusolver -lcusparse -lcurand  -lcublas -lculibos -lcudart -lpthread -ldl 
LINK_LIBS = -L$(CUDADIR)/lib64 

EXAMBLES  = examples/ex_eigs.cu examples/include/mmio.cu src/eig/*.cu src/matrix/*.cu

examples/ex_eigs:${EXAMBLES}
	${CC} ${EXAMBLES}  ${INCLUDES} ${LINK_LIBS}  ${LIBS} ${CFLAGS} -o $@ 

clean:
	-rm -f *.o examples/ex_eigs



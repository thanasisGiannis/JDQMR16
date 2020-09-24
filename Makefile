SRC=./src/
HFILES = $(SRC)*.h
CUDADIR=/usr/local/cuda-10.2

CUDA_ARCH_FLAGS := -O3 -w -arch=sm_70
CFLAGS = $(CUDA_ARCH_FLAGS)

CC = $(CUDADIR)/bin/nvcc
INCLUDES= -I$(CUDADIR)/include 
LIBS =  -lm -lcusolver -lcusparse -lcurand  -lcublas -lculibos -lcudart -lpthread -ldl 
LINK_LIBS = -L$(CUDADIR)/lib64 




# EXAMPLES COMPILATION 
EXAMBLES  = examples/ex_eigs.cu examples/include/mmio.cu src/eig/jdqmr16.cu

# LIBRARY COMPILATION
#OBJS  = 
#OBJS += 
#OBJS += 



examples/ex_eigs:${EXAMBLES}
	${CC} ${EXAMBLES}  ${INCLUDES} ${LINK_LIBS}  ${LIBS} ${CFLAGS} -o $@ 

clean:
	-rm -f *.o examples/ex_eigs


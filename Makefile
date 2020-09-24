SRC=./src/
HFILES = $(SRC)*.h
CUDADIR=/usr/local/cuda-10.2

OBJS  = examples/ex_eigs.cu
#OBJS += 
#OBJS += 
#OBJ += 

CUDA_ARCH_FLAGS := -O3 -w -arch=sm_70
CFLAGS = $(CUDA_ARCH_FLAGS)

CC = $(CUDADIR)/bin/nvcc
INCLUDES= -I$(CUDADIR)/include 
LIBS =  -lm -lcusolver -lcusparse -lcurand  -lcublas -lculibos -lcudart -lpthread -ldl 
LINK_LIBS = -L$(CUDADIR)/lib64 


examples/ex_eigs:${OBJ}
	${CC} ${OBJS}  ${INCLUDES} ${LINK_LIBS}  ${LIBS} ${CFLAGS} -o $@ 

clean:
	-rm -f *.o examples/ex_eigs


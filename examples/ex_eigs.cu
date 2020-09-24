#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <unistd.h> 
#include <curand.h>
#include <cusolverDn.h>
#include <math.h>
#include <sys/time.h>
#include "../include/jdqmr16.h"

int main(){

//	char *mName = "MTX_FILES/1138_bus.mtx"; //1138
//	char *mName = "MTX_FILES/msc04515.mtx"; //4,515
//	char *mName = "MTX_FILES/494_bus.mtx"; // 494
	char *mName = "MTX_FILES/nos4.mtx"; // 100
//	char *mName = "MTX_FILES/bcsstk01.mtx"; //48
//	char *mName = "MTX_FILES/finan512.mtx"; //74752
//	char *mName = "MTX_FILES/Andrews.mtx"; 
//	char *mName = "MTX_FILES/nd24k.mtx"; 
//	char *mName = "MTX_FILES/Lap7p1M.mtx"; 
//	char *mName = "MTX_FILES/G3_circuit.mtx"; // 1,585,478
	

	struct jdqmr16Matrix *A = (struct jdqmr16Matrix *)malloc(sizeof(struct jdqmr16Matrix));
   free(A);

   
	
	return 0;
}




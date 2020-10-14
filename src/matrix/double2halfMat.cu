#include <stdio.h>
#include <cuda_fp16.h>

#include "double2halfMat.h"

__global__
void __double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols, int BLOCK);

__global__
void __half2doubleMat(double *A,int ldA, half *A16, int ldA16, int rows, int cols, int BLOCK);


cudaError_t half2doubleMat(double *A, int ldA, half *A16, int ldA16, int rows, int cols){

	int BLOCK=1;
	int maxThreadX = 8;
	int maxThreadY = 4;//maxThreadX;

	int maxGridX = (rows)/(BLOCK*maxThreadX)+1;
	int maxGridY = (cols)/maxThreadY+1;

	dim3 gridSize = dim3(maxGridX,maxGridY);
	dim3 blockSize = dim3(maxThreadX,maxThreadY);

	__half2doubleMat<<< gridSize , blockSize >>>(A,ldA, A16,ldA16, rows, cols, BLOCK);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}



cudaError_t double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols){

   int BLOCK=1;
	
	int maxThreadX = 8;
	int maxThreadY = 4;// maxThreadX;

	int maxGridX = (rows)/(BLOCK*maxThreadX)+1;
	int maxGridY = (cols)/maxThreadY+1;

	dim3 gridSize  = dim3(maxGridX,maxGridY);
	dim3 blockSize = dim3(maxThreadX,maxThreadY);

   __double2halfMat<<< gridSize , blockSize >>>(A16, ldA16, A, ldA, rows, cols,BLOCK);

	cudaDeviceSynchronize();
	return cudaGetLastError();

}



__global__
void __double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols, int BLOCK)
{

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	i = i*BLOCK;
	if (i >= rows) return;
	if (j >= cols) return;

   float val = (float)A[i+j*ldA];
   A16[i+j*ldA16] = __float2half(val);

   return;



	for(int k=0;k<BLOCK;k++){
		if( (i+k-1) >= rows)
			break;
		float val = (float)A[(i+k) +j*ldA];
		A16[(i+k-1)+j*ldA16] = __float2half(val);
	}
	return ;
}

__global__
void __half2doubleMat(double *A,int ldA, half *A16, int ldA16, int rows, int cols, int BLOCK)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	i *= BLOCK;

	if (i >= rows) return;
	if (j >= cols) return;

   A[i + j*ldA] =(double) __half2float(A16[i + j*ldA16]);
   return;

	for(int k=0;k<BLOCK;k++){
		if( (i+k-1) >= rows)
			break;
		A[ (i+k-1) + j*ldA] =(double) __half2float(A16[(i+k) + j*ldA16]);
	}


	return ;
}



#include "../../include/jdqmr16.h"
#include "../include/helper.h"

#include <stdio.h>
#include <cuda_fp16.h>

#include "double2halfMat.h"


__global__
void __double2floatMat(float *A32, int ldA32, double *A, int ldA, int rows, int cols, int BLOCK){

   int COL = blockIdx.y*blockDim.y+threadIdx.y;
   int ROW = blockIdx.x*blockDim.x+threadIdx.x;

   if (ROW < rows && COL < cols) {
      A32[ROW + COL*ldA32] = (float)A[ROW + COL*ldA];
   }
}

__global__
void __float2doubleMat(double *A,int ldA, float *A32, int ldA32, int rows, int cols, int BLOCK){

   int COL = blockIdx.y*blockDim.y+threadIdx.y;
   int ROW = blockIdx.x*blockDim.x+threadIdx.x;

   if (ROW < rows && COL < cols) {
      A[ROW + COL*ldA] = (double)A32[ROW + COL*ldA32];
   }


}






__global__
void __double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols, int BLOCK){

   int COL = blockIdx.y*blockDim.y+threadIdx.y;
   int ROW = blockIdx.x*blockDim.x+threadIdx.x;

   if (ROW < rows && COL < cols) {
      float val = (float)A[ROW + COL*ldA];
      A16[ROW + COL*ldA16] = __float2half(val);
   }
}

__global__
void __half2doubleMat(double *A,int ldA, half *A16, int ldA16, int rows, int cols, int BLOCK){

   int COL = blockIdx.y*blockDim.y+threadIdx.y;
   int ROW = blockIdx.x*blockDim.x+threadIdx.x;

   if (ROW < rows && COL < cols) {
      float val = __half2float(A16[ROW + COL*ldA16]);
      A[ROW + COL*ldA] = (double)val;
   }


}


cudaError_t half2doubleMat(double *A, int ldA, half *A16, int ldA16, int rows, int cols){


   dim3 threadsPerBlock(rows,cols);
   dim3 blocksPerGrid(1, 1);
   if (rows*cols > 16){

      if(cols>1){
         threadsPerBlock.x = 8;
         threadsPerBlock.y = 2;
      }else{
         threadsPerBlock.x = 16;
         threadsPerBlock.y = 1;

      }
      blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(cols)/double(threadsPerBlock.y));
   }

   __half2doubleMat<<<blocksPerGrid,threadsPerBlock>>>(A,ldA, A16,ldA16,rows,cols,1);
   CUDA_CALL(cudaGetLastError());   
   return cudaGetLastError();
}



cudaError_t double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols){


   dim3 threadsPerBlock(rows,cols);
   dim3 blocksPerGrid(1, 1);
   if (rows*cols > 16){
      if(cols>1){
         threadsPerBlock.x = 8;
         threadsPerBlock.y = 2;
      }else{
         threadsPerBlock.x = 16;
         threadsPerBlock.y = 1;

      }
      blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(cols)/double(threadsPerBlock.y));
   }

   __double2halfMat<<<blocksPerGrid,threadsPerBlock>>>(A16,ldA16, A,ldA,rows,cols,1);
   CUDA_CALL(cudaGetLastError());   
   return cudaGetLastError();
}



cudaError_t float2doubleMat(double *A, int ldA, float *A32, int ldA32, int rows, int cols){


   dim3 threadsPerBlock(rows,cols);
   dim3 blocksPerGrid(1, 1);
   if (rows*cols > 16){

      if(cols>1){
         threadsPerBlock.x = 8;
         threadsPerBlock.y = 2;
      }else{
         threadsPerBlock.x = 16;
         threadsPerBlock.y = 1;

      }
      blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(cols)/double(threadsPerBlock.y));
   }

   __float2doubleMat<<<blocksPerGrid,threadsPerBlock>>>(A,ldA, A32,ldA32,rows,cols,1);
   CUDA_CALL(cudaGetLastError());   
   return cudaGetLastError();
}



cudaError_t double2floatMat(float *A32, int ldA32, double *A, int ldA, int rows, int cols){


   dim3 threadsPerBlock(rows,cols);
   dim3 blocksPerGrid(1, 1);
   if (rows*cols > 16){
      if(cols>1){
         threadsPerBlock.x = 8;
         threadsPerBlock.y = 2;
      }else{
         threadsPerBlock.x = 16;
         threadsPerBlock.y = 1;

      }
      blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
      blocksPerGrid.y = ceil(double(cols)/double(threadsPerBlock.y));
   }

   __double2floatMat<<<blocksPerGrid,threadsPerBlock>>>(A32,ldA32, A,ldA,rows,cols,1);
   CUDA_CALL(cudaGetLastError());   
   return cudaGetLastError();
}








#include "../../include/jdqmr16.h"
#include "../include/helper.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "innerSolver.h"

void innerSolver_init(double *P, int ldP, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;


   cudaMalloc((void**)&(spInnerSolver->B),sizeof(double)*dim*numEvals);
   spInnerSolver->ldB = dim;
   cudaMalloc((void**)&(spInnerSolver->VTB),sizeof(double)*numEvals*numEvals);
   spInnerSolver->ldVTB = numEvals;
   cudaMalloc((void**)&(spInnerSolver->X),sizeof(double)*numEvals*dim);
   spInnerSolver->ldX = dim;      


   spInnerSolver->maxB       = (double*)malloc(sizeof(double));
   spInnerSolver->n  ormIndexB = (int*)malloc(sizeof(int));

}

void innerSolver_destroy(struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

   cudaFree(spInnerSolver->B);
   cudaFree(spInnerSolver->VTB);
   cudaFree(spInnerSolver->X);

   free(spInnerSolver->normIndexB);
   free(spInnerSolver->maxB);
   
}

void innerSolver(double *P, int ldP, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd){
/* 
1)   Ax = (I-VV')*R/||(I-VV')*R||;
2)   P   = (I-VV')*x;

For step 1 sQMR with early stopping is used
*/

   struct gpuHandler        *gpuH          = jd->gpuH;

   cublasHandle_t         cublasH   = gpuH->cublasH;
   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

   double *B          = spInnerSolver->B;     int ldB = spInnerSolver->ldB;
   double *VTB        = spInnerSolver->VTB;   int ldVTB = spInnerSolver->ldVTB;
   double *X          = spInnerSolver->X;     int ldX = spInnerSolver->ldX;      
   double *maxB       = spInnerSolver->maxB;
   int    *normIndexB = spInnerSolver->normIndexB;



   CUDA_CALL(cudaMemset(P,0,sizeof(double)*ldP*numEvals));
   /* B = R */
   CUDA_CALL(cudaMemcpy(B,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice));
   /* VTR = V'*R */

   /* B = -V*VTR <=> B = R-V*V'*R */
   double minus_one = -1.0;
   double zero      =  0.0;
   double one       =  1.0;

   cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                           V, CUDA_R_64F,ldV,R,CUDA_R_64F,ldR,
                           &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                           CUBLAS_GEMM_ALGO2);
   

   cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                           V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                           &one,B,CUDA_R_64F,ldB,CUDA_R_64F,
                           CUBLAS_GEMM_ALGO2);


   /* normalize B with infinity norm */
   double alpha;
   for(int i=0; i<numEvals; i++){
      cublasIdamax(cublasH,dim,&B[0+i*ldP], 1, normIndexB);
      (*normIndexB)--;
      CUDA_CALL(cudaMemcpy(maxB,&B[*normIndexB + i*ldB],sizeof(double),cudaMemcpyDeviceToHost));
      alpha = 1.0/(*maxB);
      cublasDscal(cublasH,dim,&alpha,&B[0+i*ldP],1);   
   }


   /* At this point solve the numEvals systems */
   // in the near future this will be sQMR
   CUDA_CALL(cudaMemcpy(X,B,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice));

   /* P = X-V*V'*X */
   CUDA_CALL(cudaMemcpy(P,X,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice));
   cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                           V, CUDA_R_64F,ldV,R,CUDA_R_64F,ldR,
                           &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                           CUBLAS_GEMM_ALGO2);
   

   cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                           V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                           &one,P,CUDA_R_64F,ldP,CUDA_R_64F,
                           CUBLAS_GEMM_ALGO2);

}
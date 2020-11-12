
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>


#include "../matrix/double2halfMat.h"
#include "../../include/jdqmr16.h"
#include "../include/helper.h"

#include "innerSolver.h"
#include "sqmr.h"

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
   spInnerSolver->normIndexB = (int*)malloc(sizeof(int));


   // init sQMR 
   if(jd->useHalf == 1){
      // fp16 
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->X16),sizeof(half)*dim*numEvals));
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->B16),sizeof(half)*dim*numEvals));
      spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
      sqmr_init((half*)spInnerSolver->X16, dim, (half*)spInnerSolver->B16, dim, dim, 0, jd);
   }else if(jd->useHalf == 0 ){
      //fp64      
      spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
      sqmrD_init((double*)spInnerSolver->X16, dim, (double*)spInnerSolver->B16, dim, V, ldV, numEvals, dim, 0, jd);
   }else if(jd->useHalf == -1){
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->X32),sizeof(float)*dim*numEvals));
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->B32),sizeof(float)*dim*numEvals));
      spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
      //sqmrD_init((double*)spInnerSolver->X32, dim, (double*)spInnerSolver->B32, dim, V, ldV, numEvals, dim, 0, jd);
      sqmrF_init((float*)spInnerSolver->X32, dim, (float*)spInnerSolver->B32, dim, dim, 0, jd);
      }else if(jd->useHalf == -2 ){
         //fp64      
         spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
         sqmrD_init((double*)spInnerSolver->X16, dim, (double*)spInnerSolver->B16, dim, 0, 0, numEvals, dim, 0, jd);
      }

}

void innerSolver_destroy(struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

   
   if(jd->useHalf == 1){
      sqmr_destroy(jd);
   }else if(jd->useHalf == 0){
      sqmrD_destroy(jd);
   }else if(jd->useHalf == -1){
      sqmrF_destroy(jd);
   }

   free(spInnerSolver->spSQmr);

   if(jd->useHalf == -1){
      cudaFree(spInnerSolver->X32);
      cudaFree(spInnerSolver->B32);
   }else if(jd->useHalf == 1){
      cudaFree(spInnerSolver->X16);
      cudaFree(spInnerSolver->B16);

   }
   
   cudaFree(spInnerSolver->B);
   cudaFree(spInnerSolver->VTB);
   cudaFree(spInnerSolver->X);

   free(spInnerSolver->normIndexB);
   free(spInnerSolver->maxB);
   
}

void innerSolver(double *P, int ldP, double *R, int ldR, double *normr,
                  double *V, int ldV, double *L,
                  int numEvals, int dim, double tol, struct jdqmr16Info *jd){
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

   if(jd->useHalf == 1){
      /* ==== FP16 SOLVER ==== */
      /* normalize B with infinity norm */
 

      /* B = -V*VTR <=> B = R-V*V'*R */
      double minus_one = -1.0;
      double zero      =  0.0;
      double one       =  1.0;

      for(int j=0; j<numEvals; j++){
         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                                 V, CUDA_R_64F,ldV,R,CUDA_R_64F,ldR,
                                 &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));
         

         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                                 V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                                 &one,B,CUDA_R_64F,ldB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));

      }




     double alpha;
      for(int i=0; i<numEvals; i++){
         double result;
         cublasDnrm2(cublasH, dim, &B[0+i*ldP], 1, &result);
         alpha = 1.0/(result);
         cublasDscal(cublasH,dim,&alpha,&B[0+i*ldP],1);   
      }


      /* At this point solve the numEvals systems using mixed precision */
      half *X16 = (half*)spInnerSolver->X16;
      half *B16 = (half*)spInnerSolver->B16;

      CUDA_CALL(double2halfMat(B16, dim, B, ldB, dim, numEvals));
      for(int i=0;i<numEvals; i++){
         if(normr[i] < tol){
            cudaMemcpy(&X16[0+i*dim], &B16[0+i*dim], dim*sizeof(half), cudaMemcpyDeviceToDevice);
            continue;
         }
         sqmr(&X16[0+i*dim], ldX, &B16[0+i*dim], ldB, dim, 1.0, jd);
      }
      CUDA_CALL(half2doubleMat(P, ldP, X16, dim, dim, numEvals));
      /* P = X-V*V'*X */
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                              V, CUDA_R_64F,ldV,P,CUDA_R_64F,ldP,
                              &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      

      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                              V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                              &one,P,CUDA_R_64F,ldP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      return;
   }else if(jd->useHalf == 0){
      /* ==== FP64 SOLVER ==== */
      double *X_;
      double *B_;

      for(int i=0;i<numEvals; i++){
         X_ = &P[0+i*ldX];
         B_ = &B[0+i*ldB];

         if(normr[i] < tol){
            cudaMemcpy(X_, B_, dim*sizeof(double), cudaMemcpyDeviceToDevice);
            continue;
         }

         sqmrD(X_, dim, B_, dim, V, ldV, numEvals, dim, 1.0 , jd);
      }
      return;
   }else if(jd->useHalf == -1){
      /* ==== FP32 SOLVER ==== */
      /* B = -V*VTR <=> B = R-V*V'*R */
      double minus_one = -1.0;
      double zero      =  0.0;
      double one       =  1.0;

      for(int j=0; j<numEvals; j++){
         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                                 V, CUDA_R_64F,ldV,R,CUDA_R_64F,ldR,
                                 &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));
         

         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                                 V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                                 &one,B,CUDA_R_64F,ldB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));

      }


      /* normalize B with infinity norm */
      double alpha;
      for(int i=0; i<numEvals; i++){
         double result;
         cublasDnrm2(cublasH, dim, &B[0+i*ldP], 1, &result);
         alpha = 1.0/(result);
         cublasDscal(cublasH,dim,&alpha,&B[0+i*ldP],1);   
      }


      /* At this point solve the numEvals systems using mixed precision */
      float *X32 = (float*)spInnerSolver->X32;
      float *B32 = (float*)spInnerSolver->B32;

      CUDA_CALL(double2floatMat(B32, dim, B, ldB, dim, numEvals));
      for(int i=0;i<numEvals; i++){
         if(normr[i] < tol){
            cudaMemcpy(&X32[0+i*dim], &B32[0+i*dim], dim*sizeof(float), cudaMemcpyDeviceToDevice);
            continue;
         }

         sqmrF(&X32[0+i*dim], dim, &B32[0+i*dim], dim, dim, 1.0, jd);
      }
      CUDA_CALL(float2doubleMat(P, ldP, X32, dim, dim, numEvals));
      /* P = X-V*V'*X */
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                              V, CUDA_R_64F,ldV,P,CUDA_R_64F,ldR,
                              &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      

      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                              V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                              &one,P,CUDA_R_64F,ldP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      return;






   }else if(jd->useHalf == -2){
      /* ==== FP64 v2 SOLVER ==== */
      /* B = -V*VTR <=> B = R-V*V'*R */
      double minus_one = -1.0;
      double zero      =  0.0;
      double one       =  1.0;

      for(int j=0; j<numEvals; j++){
         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                                 V, CUDA_R_64F,ldV,R,CUDA_R_64F,ldR,
                                 &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));
         

         CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                                 V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                                 &one,B,CUDA_R_64F,ldB,CUDA_R_64F,
                                 CUBLAS_GEMM_DEFAULT));

      }


      /* normalize B with infinity norm */
      double alpha;
      for(int i=0; i<numEvals; i++){
         double result;
         cublasDnrm2(cublasH, dim, &B[0+i*ldP], 1, &result);
         alpha = 1.0/(result);
         cublasDscal(cublasH,dim,&alpha,&B[0+i*ldP],1);   
      }

      for(int i=0;i<numEvals; i++){
         if(normr[i] < tol){
            cudaMemcpy(&P[0+i*dim], &B[0+i*dim], dim*sizeof(double), cudaMemcpyDeviceToDevice);
            continue;
         }
         sqmrD(&P[0+i*dim], dim, &B[0+i*dim], dim, 0,0, 1,dim, 1.0, jd);
      }
      /* P = X-V*V'*X */
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                              V, CUDA_R_64F,ldV,P,CUDA_R_64F,ldR,
                              &zero,VTB,CUDA_R_64F,ldVTB,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      

      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                              V, CUDA_R_64F,ldV,VTB,CUDA_R_64F,ldVTB,
                              &one,P,CUDA_R_64F,ldP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      return;






   }
}












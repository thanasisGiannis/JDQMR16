
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
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->X16),sizeof(half)*dim));
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->B16),sizeof(half)*dim));
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->V16),sizeof(double)*dim*numEvals));spInnerSolver->ldV16=dim;
      spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
      sqmr_init((half*)spInnerSolver->X16, dim, (half*)spInnerSolver->B16, dim, dim, 0, jd);
   }else{
      //fp64      
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->X16),sizeof(double)*dim));
      CUDA_CALL(cudaMalloc((void**)&(spInnerSolver->B16),sizeof(double)*dim));
      spInnerSolver->spSQmr = (struct sqmrSpace *)malloc(sizeof(struct sqmrSpace));
      sqmrD_init((double*)spInnerSolver->X16, dim, (double*)spInnerSolver->B16, dim, V, ldV, numEvals, dim, 0, jd);
   }
}

void innerSolver_destroy(struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

   
   if(jd->useHalf == 1){
      sqmr_destroy(jd);
   }else{
      sqmrD_destroy(jd);
   }
   free(spInnerSolver->spSQmr);
   cudaFree(spInnerSolver->X16);
   cudaFree(spInnerSolver->B16);
   cudaFree(spInnerSolver->V16);

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



   if(jd->useHalf == 1){
      /* ==== FP16 SOLVER ==== */

      /* normalize B with infinity norm */
      double alpha;
      for(int i=0; i<numEvals; i++){
         cublasIdamax(cublasH,dim,&B[0+i*ldP], 1, normIndexB);
         (*normIndexB)--;
         CUDA_CALL(cudaMemcpy(maxB,&B[*normIndexB + i*ldB],sizeof(double),cudaMemcpyDeviceToHost));
         alpha = 1.0/(*maxB);
         cublasDscal(cublasH,dim,&alpha,&B[0+i*ldP],1);   
      }


      /* At this point solve the numEvals systems using mixed precision */
      half *X16 = (half*)spInnerSolver->X16;
      half *B16 = (half*)spInnerSolver->B16;


      
      for(int i=0;i<numEvals; i++){

         /* 
           create a diagonal matrix which saves the column norms
           use it along with the ritz value to scale the coefficient matrix
           cast matrix into half precision
         */


         /*
            apply diagonal matrix to the right hand side vector
            cast vectors to half precision
         */
         CUDA_CALL(double2halfMat(X16, dim, &X[0+i*ldX], ldX, dim, 1));
         CUDA_CALL(double2halfMat(B16, dim, &B[0+i*ldB], ldB, dim, 1));


         
         /* 
            sqmr should take into account the ritz vectors matrix
            so it should have a low precision matrix representation
            that feeds the sqmr() function
         */
         sqmr(X16, ldX, B16, ldB, dim, 1.0, jd);

         /* 
            cast vector into full precision
            and then apply the diagonal matrix in order to map 
            the solution to the original linear system
         */
         CUDA_CALL(half2doubleMat(&X[0+i*ldX], ldX, X16, dim, dim, 1));
      }

//      printMatrixHalf(X16,dim,1,"X");
//      printMatrixHalf(B16,dim,1,"B");

//      struct jdqmr16Matrix  *A = jd->matrix;
//      printMatrixInt(A->devRows,A->nnz,1,"rows");
//      printMatrixInt(A->devCols,A->nnz,1,"cols"); 

/*
      for(int j=0;j<A->nnz;j++){
         double *vD = A->devValuesD;
         half   *vH = A->devValuesH;
         printMatrixHalf(&vH[j],1,1,"vH");
         printMatrixDouble(&vD[j],1,1,"vD");
         printf("--------\n");
      }
 
*/

      /* This should not be in use after changes */
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

   }else{
      /* ==== FP64 SOLVER ==== */
      double *X_ = (double*)spInnerSolver->X16;
      double *B_ = (double*)spInnerSolver->B16;
//      for(int i=0;i<numEvals; i++){
      for(int i=0;i<1; i++){

         CUDA_CALL(cudaMemcpy(X_, &X[0+i*ldX], dim*sizeof(double), cudaMemcpyDeviceToDevice));
         CUDA_CALL(cudaMemcpy(B_, &B[0+i*ldB], dim*sizeof(double), cudaMemcpyDeviceToDevice));
         sqmrD(X_, dim, B_, dim, V, ldV, numEvals, dim, 1.0 , jd);
/*
         printMatrixDouble(X_,dim,1,"X");
         printMatrixDouble(B_,dim,1,"B");
         printMatrixDouble(V,dim,numEvals,"V");
         struct jdqmr16Matrix  *A = jd->matrix;
         printMatrixDouble(A->devValuesD,A->nnz,1,"vals");
         printMatrixInt(A->devRows,A->nnz,1,"rows");
         printMatrixInt(A->devCols,A->nnz,1,"cols");  
*/
         CUDA_CALL(cudaMemcpy(&X[0+i*ldX], X_, dim*sizeof(double), cudaMemcpyDeviceToDevice));
   }

      CUDA_CALL(cudaMemcpy(P,X,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice));
   }
}


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


   innerSolverSpace *spInnerSolver = jd->spInnerSolver;
   cudaMalloc((void**)&(spInnerSolver->B),sizeof(double)*dim*numEvals);

   spInnerSolver->spBlQmr = (blQmrSpace*)malloc(sizeof(blQmrSpace));

   switch(jd->useHalf){

      case USE_FP16:
         cudaMalloc((void**)&(spInnerSolver->B32),sizeof(float)*dim*numEvals); spInnerSolver->ldB32 = dim;
         cudaMalloc((void**)&(spInnerSolver->P32),sizeof(float)*dim*numEvals); spInnerSolver->ldP32 = dim;
         cudaMalloc((void**)&(spInnerSolver->V32),sizeof(float)*dim*numEvals); spInnerSolver->ldV32 = dim;
         blQmrH_init((float*)(spInnerSolver->P32), spInnerSolver->ldP32, (float*)(spInnerSolver->B32), spInnerSolver->ldB32,
                        (float*)(spInnerSolver->V32), spInnerSolver->ldV32,dim, numEvals,numEvals,
                        0.0, 0, spInnerSolver->spBlQmr,jd);
         blQmrD_init(P, ldP, R, ldR, V, ldV, dim, numEvals,numEvals, 0.0, 0, spInnerSolver->spBlQmr,jd);
         break;

      case USE_FP32:
         cudaMalloc((void**)&(spInnerSolver->B32),sizeof(float)*dim*numEvals); spInnerSolver->ldB32 = dim;
         cudaMalloc((void**)&(spInnerSolver->P32),sizeof(float)*dim*numEvals); spInnerSolver->ldP32 = dim;
         cudaMalloc((void**)&(spInnerSolver->V32),sizeof(float)*dim*numEvals); spInnerSolver->ldV32 = dim;
         blQmrF_init((float*)(spInnerSolver->P32), spInnerSolver->ldP32, (float*)(spInnerSolver->B32), spInnerSolver->ldB32,
                        (float*)(spInnerSolver->V32), spInnerSolver->ldV32,dim, numEvals,numEvals,
                        0.0, 0, spInnerSolver->spBlQmr,jd);
         blQmrD_init(P, ldP, R, ldR, V, ldV, dim, numEvals,numEvals, 0.0, 0, spInnerSolver->spBlQmr,jd);
         break;

       default:
         blQmrD_init(P, ldP, R, ldR, V, ldV, dim, numEvals,numEvals, 0.0, 0, spInnerSolver->spBlQmr,jd);
         return;
   }

}

void innerSolver_destroy(struct jdqmr16Info *jd){


   innerSolverSpace *spInnerSolver = jd->spInnerSolver;
   cudaFree(spInnerSolver->B);
   switch (jd->useHalf){
       case USE_FP16:
         cudaFree(spInnerSolver->B32);
         cudaFree(spInnerSolver->P32);
         cudaFree(spInnerSolver->V32);
         blQmrH_destroy(jd->spInnerSolver->spBlQmr);
         blQmrD_destroy(jd->spInnerSolver->spBlQmr);
         break;

       case USE_FP32:
         cudaFree(spInnerSolver->B32);
         cudaFree(spInnerSolver->P32);
         cudaFree(spInnerSolver->V32);
         blQmrF_destroy(jd->spInnerSolver->spBlQmr);
         blQmrD_destroy(jd->spInnerSolver->spBlQmr);
         break;

       default:
         blQmrD_destroy(jd->spInnerSolver->spBlQmr);
         return;
   }
   free(spInnerSolver->spBlQmr);

}

void innerSolver(double *P, int ldP, double *R, int ldR, double *normr,
                  double *V, int ldV, double *L,
                  int numEvals, int dim, double tol, struct jdqmr16Info *jd){

   // P = inv((I-VV')M(I-VV'))*R
   innerSolverSpace *spInnerSolver = jd->spInnerSolver;


   double *B = spInnerSolver->B;
   
   float *B32 = (float *)spInnerSolver->B32;
   float *P32 = (float *)spInnerSolver->P32;
   float *V32 = (float *)spInnerSolver->V32;
   
   int ldB32 = spInnerSolver->ldB32;
   int ldP32 = spInnerSolver->ldP32;
   int ldV32 = spInnerSolver->ldV32;


   int numNotConverged = 0;
   int numConverged    = 0;

   // Check wich eigenvectors have converged
   // the ones that converged are put in the end of P
   // the rest are put in the start of Β
   // then Β(:,0:numNotConverged) is used in the block qmr method
   int pivotThitaIdx = -1;
   double normA = jd->normMatrix;
   for(int j=0; j<numEvals; j++){
      if(normr[j] > tol*normA){
         // j-vector is not converged
         cudaMemcpy(&B[0+numNotConverged*ldP],&R[0+j*ldR],sizeof(double)*dim,cudaMemcpyDeviceToDevice);
         numNotConverged++;

         if (pivotThitaIdx == -1){
            pivotThitaIdx = j;
         }
      }else{
         // j-vector is converged
         cudaMemcpy(&P[0+(numEvals-1-numConverged)*ldP],&R[0+j*ldR],sizeof(double)*dim,cudaMemcpyDeviceToDevice);
         
         numConverged++;
      }

   }


   if(numConverged == numEvals)
      return;

   if(numNotConverged < 3){
      blQmrD(P, ldP, B, dim, V, ldV, dim, numNotConverged, numEvals,
          pivotThitaIdx, tol, 3*dim, jd->spInnerSolver->spBlQmr,jd);
      return;
   }
   /* here to be set the sqmr block method */
   double scalB;
   switch(jd->useHalf){
      case USE_FP16:
         for(int j=0; j<numEvals; j++){
            cublasDnrm2(jd->gpuH->cublasH,dim,&B[0+j*dim], 1, &scalB);
            scalB = 1.0/(scalB);
            cublasDscal(jd->gpuH->cublasH, dim,&scalB,&B[0+j*dim], 1);
         }
         double2floatMat(B32, dim, B, dim, dim, numNotConverged);
         double2floatMat(V32, dim, V, dim, dim, numEvals);
         // at this point a fp32 solver is utilized
         // Inside the solver chooses fp32 or fp16 for the matvecs with coefficient matrix
         
         blQmrH(P32, ldP32, B32, dim, V32, ldV32, dim, numNotConverged, numEvals, pivotThitaIdx,
               tol, 3*dim, jd->spInnerSolver->spBlQmr,jd);
         float2doubleMat(P, dim, P32, dim, dim, numNotConverged);
         break;
      case USE_FP32:
         for(int j=0; j<numEvals; j++){
            cublasDnrm2(jd->gpuH->cublasH,dim,&B[0+j*dim], 1, &scalB);
            scalB = 1.0/(scalB);
            cublasDscal(jd->gpuH->cublasH, dim,&scalB,&B[0+j*dim], 1);
         }
         double2floatMat(B32, dim, B, dim, dim, numNotConverged);
         double2floatMat(V32, dim, V, dim, dim, numEvals);
         // at this point a fp32 solver is utilized
         // Inside the solver chooses fp32 or fp16 for the matvecs with coefficient matrix
         //blQmrF(P32, ldP32, B32, dim, V32, ldV32, dim, numNotConverged, numEvals, tol, 10*dim, jd->spInnerSolver->spBlQmr,jd);
         blQmrF(P32, ldP32, B32, dim, V32, ldV32, dim, numNotConverged, numEvals,  pivotThitaIdx, 
                  tol, 3*dim, jd->spInnerSolver->spBlQmr,jd);

         float2doubleMat(P, dim, P32, dim, dim, numNotConverged);
        break;

      default:
         // at this point a fp64 block sqmr is utilized to solve the problem 
         blQmrD(P, ldP, B, dim, V, ldV, dim, numNotConverged, numEvals, pivotThitaIdx, tol, 3*dim, jd->spInnerSolver->spBlQmr,jd);
   }

}












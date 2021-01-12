
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

   switch(jd->useHalf){
      case USE_FP32 || USE_FP16:
         cudaMalloc((void**)&(spInnerSolver->B32),sizeof(float)*dim*numEvals);
         cudaMalloc((void**)&(spInnerSolver->P32),sizeof(float)*dim*numEvals);
         cudaMalloc((void**)&(spInnerSolver->V32),sizeof(float)*dim*numEvals);
         break;

       default:
         return;
   }

}

void innerSolver_destroy(struct jdqmr16Info *jd){

   innerSolverSpace *spInnerSolver = jd->spInnerSolver;
   cudaFree(spInnerSolver->B);
   
   switch (jd->useHalf){
       case USE_FP16:
         cudaFree(spInnerSolver->B16);
         cudaFree(spInnerSolver->P16);
         cudaFree(spInnerSolver->V16);
         break;

       case USE_FP32:
         cudaFree(spInnerSolver->B32);
         cudaFree(spInnerSolver->P32);
         cudaFree(spInnerSolver->V32);
         break;

       default:
         return;
   }

}

void innerSolver(double *P, int ldP, double *R, int ldR, double *normr,
                  double *V, int ldV, double *L,
                  int numEvals, int dim, double tol, struct jdqmr16Info *jd){

   // P = inv((I-VV')M(I-VV'))*R
   innerSolverSpace *spInnerSolver = jd->spInnerSolver;


   double *B = spInnerSolver->B;
   half *B16 = (half *)spInnerSolver->B16;
   half *P16 = (half *)spInnerSolver->P16;

   float *B32 = (float *)spInnerSolver->B32;
   float *P32 = (float *)spInnerSolver->P32;

   int numNotConverged = 0;
   int numConverged    = 0;

   // Check wich eigenvectors have converged
   // the ones that converged are put in the end of P
   // the rest are put in the start of Β
   // then Β(:,0:numNotConverged) is used in the block qmr method
   double normA = jd->normMatrix;
   for(int j=0; j<numEvals; j++){
      if(normr[j] > tol*normA){
         // j-vector is not converged
         cudaMemcpy(&B[0+numNotConverged*ldP],&R[0+j*ldR],sizeof(double)*dim,cudaMemcpyDeviceToDevice);
         numNotConverged++;
      }else{
         // j-vector is converged
         cudaMemcpy(&P[0+(numEvals-1-numConverged)*ldP],&R[0+j*ldR],sizeof(double)*dim,cudaMemcpyDeviceToDevice);
         numConverged++;
      }

   }


   if(numConverged == numEvals)
      return;

   /* here to be set the sqmr block method */
   double scalB;
   switch(jd->useHalf){
      case USE_FP32 || USE_FP16:
         for(int j=0; j<numEvals; j++){
            cublasDnrm2(jd->gpuH->cublasH,dim,&B[0+j*dim], 1, &scalB);
            scalB = 2048.0/(scalB);
            cublasDscal(jd->gpuH->cublasH, dim,&scalB,&B[0+j*dim], 1);
         }
         double2floatMat(B32, dim, B, dim, dim, numNotConverged);
         // at this point a fp32 solver is utilized
         // Inside the solver chooses fp32 or fp16 for the matvecs with coefficient matrix
         cudaMemcpy(P32,B32,sizeof(float)*dim*numNotConverged,cudaMemcpyDeviceToDevice);

         float2doubleMat(P, dim, P32, dim, dim, numNotConverged);
        break;

      default:
         // at this point a fp64 block sqmr is utilized to solve the problem 
         cudaMemcpy(P,B,sizeof(double)*dim*numNotConverged,cudaMemcpyDeviceToDevice);
   }

}












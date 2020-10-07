#include "../../include/jdqmr16.h"
#include "../include/helper.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "initBasis.h"
#include "eigH.h"
#include "residual.h"   
#include "expandBasis.h"
#include "restart.h"

void init_jdqmr16(struct jdqmr16Info *jd){
   
   /* allocate gpu memory */
   struct jdqmr16Matrix *A = jd->matrix;   
   
   double *vals    = A->values;
   int    *rows    = A->rows;
   int    *cols    = A->cols;

   double *devVals  = A->devValuesD;
   int    *devRows  = A->devRows;
   int    *devCols  = A->devCols;
   int     nnz      = A->nnz;
   int     dim      = A->dim;
   int     numEvals = jd->numEvals;
   int     maxBasis = jd->maxBasis;

  
   /* if matrix is small */
   if(numEvals*maxBasis >= dim){
      jd->maxBasis = floor(dim/numEvals);
      maxBasis = jd->maxBasis;
   }
 
   /* initialize data to device */
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesD),sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devCols),sizeof(int)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devRows),sizeof(int)*nnz));

   CUDA_CALL(cudaMemcpy((void*)(A->devValuesD),(void*)vals,sizeof(double)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)(A->devCols),(void*)cols,sizeof(int)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)(A->devRows),(void*)rows,sizeof(int)*nnz,cudaMemcpyHostToDevice));

   /* allocate device memory for solver */
   jd->sp   = (struct devSolverSpace*)malloc(sizeof(struct devSolverSpace));
   jd->gpuH = (struct gpuHandler*)malloc(sizeof(struct gpuHandler));
   
   /* initialize gpu handlers */
   struct gpuHandler *gpuH = jd->gpuH;   
   // curand
   curandGenerator_t *curandH = &(gpuH->curandH);
	curandCreateGenerator(curandH, CURAND_RNG_PSEUDO_DEFAULT);
   // cusolver
   cusolverDnHandle_t *cusolverH = &(gpuH->cusolverH);
   cusolverDnCreate(cusolverH);
   // cublas
   cublasHandle_t *cublasH =  &(gpuH->cublasH);
   cublasCreate(cublasH);   
   // cusparse
   cusparseHandle_t *cusparseH = &(gpuH->cusparseH);
   cusparseCreate(cusparseH);

   /* initialize space for solver */
   struct devSolverSpace* sp = jd->sp;
   CUDA_CALL(cudaMalloc((void**)&sp->W,sizeof(double)*dim*maxBasis*numEvals));               sp->ldW     = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->H,sizeof(double)*maxBasis*numEvals*maxBasis*numEvals)); sp->ldH     = maxBasis*numEvals;
   CUDA_CALL(cudaMalloc((void**)&sp->Vprev,sizeof(double)*numEvals*dim));                    sp->ldVprev = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->V,sizeof(double)*numEvals*dim));                        sp->ldV     = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->Lprev,sizeof(double)*numEvals)); 
   CUDA_CALL(cudaMalloc((void**)&sp->L,sizeof(double)*numEvals)); 
   CUDA_CALL(cudaMalloc((void**)&sp->R,sizeof(double)*numEvals*dim));                        sp->ldR     = dim;

   CUDA_CALL(cudaMalloc((void**)&sp->AW,sizeof(double)*maxBasis*numEvals*dim));              sp->ldAW    = dim;
   //CUDA_CALL(cudaMalloc((void**)&sp->AP,sizeof(double)*numEvals*dim));                       sp->ldAP    = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->P,sizeof(double)*numEvals*dim));                        sp->ldP     = dim;
   

   double *H        = sp->H;        /* projected Matrix */
   double *V        = sp->V;        /* Ritz vectors */
   double *W        = sp->L;        /* Ritz values */
   
   /* init inner functions */

   // init initBasis
   jd->spInitBasis = (struct initBasisSpace *)malloc(sizeof(struct initBasisSpace));
   initBasis_init(sp->W,sp->ldW, sp->H, sp->ldH, sp->V, sp->ldV,sp->L, dim, maxBasis,numEvals,jd);

   // init eigH
   jd->spEigH = (struct eigHSpace *)malloc(sizeof(struct eigHSpace));   
   eigH_init(sp->W, sp->ldW, sp->L, sp->H, sp->ldH, numEvals, maxBasis, jd);
   
   // init residual
   jd->spResidual = (struct residualSpace *)malloc(sizeof(struct residualSpace));
   residual_init(sp->R,sp->ldR,sp->V,sp->ldV,sp->L,numEvals,jd);

   // init expandBasis
   jd->spExpandBasis = (struct expandBasisSpace *)malloc(sizeof(struct expandBasisSpace));
   expandBasis_init(sp->W, sp->ldW, sp->H, sp->ldH, sp->P, sp->ldP,maxBasis, dim, numEvals, jd);

   // init restart
   jd->spRestart = (struct restartSpace *)malloc(sizeof(struct restartSpace));
   restart_init(sp->W, sp->ldW, sp->H, sp->ldH, sp->Vprev, sp->ldVprev, sp->Lprev, sp->V, sp->ldV, sp->L,
               &maxBasis, maxBasis, numEvals, dim, jd);

   return;
}

void destroy_jdqmr16(struct jdqmr16Info *jd){

   /* destroy inner functions */
   restart_destroy(jd);
   expandBasis_destroy(jd);
   initBasis_destroy(jd);
   eigH_destroy(jd);
   residual_destroy(jd);

   /* destroy gpu handlers */ 
   struct gpuHandler *gpuH = jd->gpuH;   

   // curand     
   curandGenerator_t curandH = gpuH->curandH;
	curandDestroyGenerator(curandH);
   // cusolver
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
	cusolverDnDestroy(cusolverH);
	// cublas
   cublasHandle_t cublasH = gpuH->cublasH;
   cublasDestroy(cublasH);
   // cusparse
   cusparseHandle_t cusparseH = gpuH->cusparseH;
   cusparseDestroy(cusparseH);

   /* Destroy Matrix */
   struct jdqmr16Matrix *A = jd->matrix;   
   
   double *devVals = A->devValuesD;
   int    *devRows = A->devRows;
   int    *devCols = A->devCols;
   int    nnz      = A->nnz;
   int    dim      = A->dim;

   CUDA_CALL(cudaFree(devVals));
   CUDA_CALL(cudaFree(devCols));
   CUDA_CALL(cudaFree(devRows));


   struct devSolverSpace *sp = jd->sp;


   CUDA_CALL(cudaFree(sp->AW));
   CUDA_CALL(cudaFree(sp->P));
   //CUDA_CALL(cudaFree(sp->AP));
   CUDA_CALL(cudaFree(sp->R));
   CUDA_CALL(cudaFree(sp->W));
   CUDA_CALL(cudaFree(sp->H));
   CUDA_CALL(cudaFree(sp->Vprev));
   CUDA_CALL(cudaFree(sp->V));
   CUDA_CALL(cudaFree(sp->L));
   CUDA_CALL(cudaFree(sp->Lprev));

   free(jd->spRestart);
   free(jd->spExpandBasis);
   free(jd->spInitBasis);
   free(jd->spEigH);
   free(jd->spResidual);

   free(jd->gpuH);
   free(jd->sp);
   
   return;

}

void jdqmr16(struct jdqmr16Info *jd){

   /* Generalized Davidson Iteration */
   struct jdqmr16Matrix *A = jd->matrix;   

   struct devSolverSpace* sp = jd->sp;
   double *W  = sp->W;  int ldW = sp->ldW; /* GD basis */
   double *H  = sp->H;  int ldH = sp->ldH; /* projected Matrix */
   double *V  = sp->V;  int ldV = sp->ldV; /* Ritz vectors */
   double *Vprev  = sp->Vprev;  int ldVprev = sp->ldVprev; /* Ritz vectors */
   
   double *L      = sp->L;                     /* Ritz values */
   double *Lprev  = sp->Lprev;                 /* Ritz values */
   
   double *R  = sp->R;  int ldR = sp->ldR; /* Ritz vectors */
   double *P  = sp->P;  int ldP = sp->ldP;
   double *AW = sp->AW; int ldAW = sp->ldAW;

   int     dim      = A->dim;       /* dimension of the problem */
   int     numEvals = jd->numEvals; /* number of wanted eigenvalues */
   int     maxBasis = jd->maxBasis; /* maximum size of GD */
   int     maxIter  = jd->maxIter;  /* number of maximum iterations of GD */
   int     basisSize = 1;           /* basis size in blocks */

   // Step 0.1: Initialize matrices and basis
   initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,maxBasis,numEvals,jd); // basis initilization and H creation

   // Step 0.2: First approximation of eigenpairs
   eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors
   // Step 0.3: Residual calculation
   residual(R, ldR, V, ldV, L, numEvals, jd); 
   

   /* main loop of JDQMR */
   for(int i=0;i<maxIter;i++){   

      /* Inner sQMR16 to be used here in the future */
      cudaMemcpy(P,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

      if(basisSize == maxBasis){
  
//         printMatrixDouble(Vprev,ldVprev,numEvals,"Vprev");
//         printMatrixDouble(V,ldV,numEvals,"V");
         restart(W, ldW, H, ldH, Vprev, ldVprev, Lprev,
                  V, ldV, L, &basisSize, maxBasis, numEvals, dim, jd);
//         printMatrixDouble(W,ldW,basisSize*numEvals,"W");   
//         printMatrixDouble(H,ldH,ldH,"H");
         //return;
//         break;
         printMatrixDouble(L,numEvals,1,"L");
//         printf("Restart\n");
      }

      /* Enrich basis with new vectors*/    
      expandBasis(W, ldW, H, ldH, P, ldP, AW, ldAW, basisSize, dim,numEvals, jd);
      basisSize++;

      /* keep previous ritz vectors for restarting purposes*/
      cudaMemcpy(Vprev,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      cudaMemcpy(Lprev,L,sizeof(double)*numEvals,cudaMemcpyDeviceToDevice);
      /* Find new Ritz pairs */
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors
      
      /* Residual calculation */
      residual(R, ldR, V, ldV, L, numEvals, jd); 

      
   }



#if 1
//if(i>3){
//   printMatrixDouble(V,ldV,numEvals,"V");
   printMatrixDouble(L,numEvals,1,"L");
 //  printMatrixDouble(Vprev,ldVprev,numEvals,"Vprev");
//   printMatrixDouble(Lprev,numEvals,1,"Lprev");
   //printMatrixDouble(R,ldR,numEvals,"R");
//   printMatrixDouble(W,ldW,basisSize*numEvals,"W");
//   printMatrixDouble(H,ldH,ldH,"H");

//}
#endif
         


}

































#include "../../include/jdqmr16.h"
#include "../include/helper.h"
#include "../matrix/double2halfMat.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "initBasis.h"
#include "eigH.h"
#include "residual.h"   
#include "expandBasis.h"
#include "restart.h"
#include "innerSolver.h"

void init_jdqmr16(struct jdqmr16Info *jd){
   
   /* allocate gpu memory */
   struct jdqmr16Matrix *A = jd->matrix;   
   
   double *vals    = A->values;
   int    *rows    = A->rows;
   int    *cols    = A->cols;

   //double *devVals  = A->devValuesD;
   //int    *devRows  = A->devRows;
   //int    *devCols  = A->devCols;
   int     nnz      = A->nnz;
   int     dim      = A->dim;
   int     numEvals = jd->numEvals;
   int     maxBasis = jd->maxBasis;

   jd->outerIterations = 0;
   jd->innerIterations = 0;

   if(jd->useHalf !=0){
      jd->useHalf = 1;
   }
   /* if matrix is small */
   if(numEvals*maxBasis >= dim){
      jd->maxBasis = floor(dim/numEvals);
      maxBasis = jd->maxBasis;
   }
 
   /* initialize data to device */
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesD),sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesH),sizeof(half)*nnz));

   CUDA_CALL(cudaMalloc((void**)&(A->devCols),sizeof(int)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devRows),sizeof(int)*nnz));

   CUDA_CALL(cudaMemcpy((void*)(A->devValuesD),(void*)vals,sizeof(double)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)(A->devCols),(void*)cols,sizeof(int)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)(A->devRows),(void*)rows,sizeof(int)*nnz,cudaMemcpyHostToDevice));



   //CUBLAS_CALL(cublas);

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
   CUDA_CALL(cudaMalloc((void**)&sp->L,sizeof(double)*numEvals)); 
   CUDA_CALL(cudaMalloc((void**)&sp->R,sizeof(double)*numEvals*dim));                        sp->ldR     = dim;

   CUDA_CALL(cudaMalloc((void**)&sp->AW,sizeof(double)*maxBasis*numEvals*dim));              sp->ldAW    = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->P,sizeof(double)*numEvals*dim));                        sp->ldP     = dim;
   

   //double *H        = sp->H;        /* projected Matrix */
   //double *V        = sp->V;        /* Ritz vectors */
   //double *W        = sp->L;        /* Ritz values */
   
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
   restart_init(sp->W, sp->ldW, sp->H, sp->ldH, sp->Vprev, sp->ldVprev, NULL , sp->V, sp->ldV, sp->L,
               &maxBasis, maxBasis, numEvals, dim, jd);

   // init innerSolver
   jd->spInnerSolver = (struct innerSolverSpace*)malloc(sizeof(struct innerSolverSpace));
   innerSolver_init(sp->P, sp->ldP, sp->R, sp->ldR, sp->V, sp->ldV, sp->L, numEvals, dim,jd);


   /*
      find norm of matrix 
      needed for termination criteria      
   */

   jd->normMatrix = 0;
   double *val = A->values;
   for(int i=0; i<A->nnz; i++){
      if(abs(val[i]) > jd->normMatrix){
         jd->normMatrix = abs(val[i]);      
      }
   }


   return;
}

void destroy_jdqmr16(struct jdqmr16Info *jd){

   /* destroy inner functions */
   //sqrm_destroy(jd);
   innerSolver_destroy(jd);
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
   
   double *devVals  = A->devValuesD;
   half   *devValsH = A->devValuesH;
   int    *devRows  = A->devRows;
   int    *devCols  = A->devCols;
   ///int    nnz       = A->nnz;
   //int    dim       = A->dim;

   CUDA_CALL(cudaFree(devVals));
   CUDA_CALL(cudaFree(devValsH));
   CUDA_CALL(cudaFree(devCols));
   CUDA_CALL(cudaFree(devRows));


   struct devSolverSpace *sp = jd->sp;


   CUDA_CALL(cudaFree(sp->AW));
   CUDA_CALL(cudaFree(sp->P));
   CUDA_CALL(cudaFree(sp->R));
   CUDA_CALL(cudaFree(sp->W));
   CUDA_CALL(cudaFree(sp->H));
   CUDA_CALL(cudaFree(sp->Vprev));
   CUDA_CALL(cudaFree(sp->V));
   CUDA_CALL(cudaFree(sp->L));

   free(jd->spInnerSolver);
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
   
   double *R  = sp->R;  int ldR = sp->ldR; /* Ritz vectors */
   double *P  = sp->P;  int ldP = sp->ldP;
   double *AW = sp->AW; int ldAW = sp->ldAW;

   int     dim       = A->dim;         /* dimension of the problem */
   int     numEvals  = jd->numEvals;   /* number of wanted eigenvalues */
   int     maxBasis  = jd->maxBasis;   /* maximum size of GD */
   int     maxIter   = jd->maxIter;    /* number of maximum iterations of GD */
   int     basisSize = 1;              /* basis size in blocks */
   double  tol       = jd->tol;        /* tolerance of convergence */
   double  normA     = jd->normMatrix; /* norm of sparse matrix */
   //double  maxerr;
   //int     iter      = 0;

   jd->numMatVecsfp64 = 0;
   jd->numMatVecsfp16 = 0;

   double *normr = (double*)malloc(sizeof(double)*numEvals);

   // Step 0.1: Initialize matrices and basis
   initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,maxBasis,numEvals,jd); // basis initilization and H creation

   // Step 0.2: First approximation of eigenpairs
   eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors
   // Step 0.3: Residual calculation
   residual(R, ldR, V, ldV, L, numEvals, jd); 
   
   /* main loop of JDQMR */
   for(int i=0;i<maxIter;i++){   
      /* Inner sQMR16 to be used here in the future */
      //cudaMemcpy(P,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      innerSolver(P,ldP,R,ldR,V,ldV,L,numEvals,dim,jd);

      if(basisSize == maxBasis){
         /* no space left - Restart basis */
         restart(W, ldW, H, ldH, Vprev, ldVprev, NULL ,
                  V, ldV, L, AW, ldAW, &basisSize, maxBasis, numEvals, dim, jd);
      }

      /* Enrich basis with new vectors*/    
      expandBasis(W, ldW, H, ldH, P, ldP, AW, ldAW, basisSize, dim,numEvals, jd);
      //basisSize++;
      
      /* keep previous ritz vectors for restarting purposes*/
      cudaMemcpy(Vprev,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      /* Find new Ritz pairs */
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors

      /* Residual calculation */
      residual(R, ldR, V, ldV, L, numEvals, jd); 

      /* convergence check */
      int    numConverged = 0;
      for(int j=0;j<numEvals;j++){
         cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
         if(normr[j] < tol*normA){
            numConverged++;
         }
      }

      #if 1
      for(int i=0;i<numEvals;i++){
         printf("||R[:,%d]||: %e\n",i,normr[i]/normA);
      }
      #endif         

      if(numConverged == numEvals){
         break;
      }
         

      jd->outerIterations++;
         
   }


   /* Get eigenpairs back */

#if 0
for(int i=0;i<numEvals;i++){
   printf("||R[:,%d]||: %e\n",i,normr[i]);
}
printMatrixDouble(L,numEvals,1,"L");

printf("Iterations=%d \nTolerance=%e\nnormA=%e\n",iter,tol,normA);
printf("fp64 matVecs=%d\nfp16 matVecs=%d\n",jd->numMatVecsfp64,jd->numMatVecsfp16);
#endif         


}



void jdqmr16_eigenpairs(double *V, int ldV, double *L, struct jdqmr16Info *jd){
   
   struct devSolverSpace* sp = jd->sp;

   cudaMemcpy(V,sp->V,sizeof(double)*(jd->numEvals)*ldV,cudaMemcpyDeviceToHost);
   cudaMemcpy(L,sp->L,sizeof(double)*(jd->numEvals),cudaMemcpyDeviceToHost);


}































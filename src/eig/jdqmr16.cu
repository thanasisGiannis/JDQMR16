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
#include "locking.h"


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

   jd->outerIterations = 0;
   jd->innerIterations = 0;

   if(jd->useHalf !=-1 && jd->useHalf != 0 && jd->useHalf!=1 && jd->useHalf!=-2){
      jd->useHalf = 1;
   }
   /* if matrix is small */
   if(numEvals*maxBasis >= dim){
      jd->maxBasis = floor(dim/numEvals);
      maxBasis = jd->maxBasis;
   }
 
   /* initialize data to device */
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesD),sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesF),sizeof(double)*nnz));
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
   cublasSetMathMode(*cublasH, CUBLAS_TENSOR_OP_MATH);
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

   CUDA_CALL(cudaMalloc((void**)&sp->Qlocked,sizeof(double)*numEvals*dim));                  sp->ldQlocked= dim;
   CUDA_CALL(cudaMalloc((void**)&sp->Llocked,sizeof(double)*numEvals)); 
   sp->numLocked = 0;

   CUDA_CALL(cudaMalloc((void**)&sp->AW,sizeof(double)*maxBasis*numEvals*dim));              sp->ldAW    = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->P,sizeof(double)*numEvals*dim));                        sp->ldP     = dim;
   
   sp->normr = (double*)malloc(sizeof(double)*numEvals);
   double *H        = sp->H;        /* projected Matrix */
   double *V        = sp->V;        /* Ritz vectors */
   double *W        = sp->L;        /* Ritz values */
   
   /* init inner functions */
   jd->gpuMemSpaceDoubleSize=0;
   jd->gpuMemSpaceIntSize=0;
   jd->gpuMemSpaceVoidSize=0;
   // init initBasis /* GPU Global Mem Done */
   jd->spInitBasis = (struct initBasisSpace *)malloc(sizeof(struct initBasisSpace));
   initBasis_init(sp->W,sp->ldW, sp->H, sp->ldH, sp->V, sp->ldV,sp->L, dim, maxBasis,numEvals,jd);

   // init eigH /* GPU Global Mem Done */
   jd->spEigH = (struct eigHSpace *)malloc(sizeof(struct eigHSpace));   
   eigH_init(sp->W, sp->ldW, sp->L, sp->H, sp->ldH, numEvals, maxBasis, jd);
   
   // init residual /* GPU Global Mem Done */
   jd->spResidual = (struct residualSpace *)malloc(sizeof(struct residualSpace));
   residual_init(sp->R,sp->ldR,sp->V,sp->ldV,sp->L,numEvals,jd);

   // init expandBasis /* GPU Global Mem Done */
   jd->spExpandBasis = (struct expandBasisSpace *)malloc(sizeof(struct expandBasisSpace));
   expandBasis_init(sp->W, sp->ldW, sp->H, sp->ldH, sp->P, sp->ldP,maxBasis, dim, numEvals, jd);

   // init restart /* GPU Global Mem Done */
   jd->spRestart = (struct restartSpace *)malloc(sizeof(struct restartSpace));
   restart_init(sp->W, sp->ldW, sp->H, sp->ldH, sp->Vprev, sp->ldVprev, NULL , sp->V, sp->ldV, sp->L,
               &maxBasis, maxBasis, numEvals, dim, jd);

   // init innerSolver
   jd->spInnerSolver = (struct innerSolverSpace*)malloc(sizeof(struct innerSolverSpace));
   innerSolver_init(sp->P, sp->ldP, sp->R, sp->ldR, sp->V, sp->ldV, sp->L, numEvals, dim,jd);

   // init locking
   jd->spLock = (struct lockSpace*)malloc(sizeof(struct lockSpace));
   sp->maxLockedVals = 2*numEvals;
   lock_init(sp->V, sp->ldV, sp->L, sp->R, sp->ldR, NULL,sp->Qlocked, sp->ldQlocked,
             sp->Llocked,sp->W,sp->ldW,sp->H,sp->ldH,sp->AW,sp->ldAW, 
            sp->numLocked, numEvals, maxBasis, sp->numLocked, dim, 1e-08,jd);
   if(jd->locking != 1 && jd->locking != 0 ){
      jd->locking = 0;   
   }


   /* Allocate Global GPU Memory to be used by all inner functions */
   cudaMalloc((void**)&(jd->gpuMemSpaceDouble),jd->gpuMemSpaceDoubleSize*sizeof(double));
   cudaMalloc((void**)&(jd->gpuMemSpaceInt),jd->gpuMemSpaceIntSize*sizeof(int));
   cudaMalloc((void**)&(jd->gpuMemSpaceVoid),jd->gpuMemSpaceVoidSize);

   /* find norm of matrix */
   jd->normMatrix = 0;
   double *val = A->values;
   for(int i=0; i<A->nnz; i++){
      if(abs(val[i]) > jd->normMatrix){
         jd->normMatrix = abs(val[i]);      
      }
   }

   jd->alpha = 1.0;
   if(jd->useHalf == 1){
      /* Half precision matrix creation */
      double *vec; cudaMalloc((void**)&vec,(A->nnz)*sizeof(double));
      cudaMemcpy(vec,A->devValuesD,(A->nnz)*sizeof(double),cudaMemcpyDeviceToDevice);
      double alpha; 

      if(jd->normMatrix > 5e+03 || jd->normMatrix < 5e-03){
         alpha = 2048.0/(jd->normMatrix);
         cublasScalEx(*cublasH,A->nnz,&alpha,CUDA_R_64F,vec,CUDA_R_64F,1,CUDA_R_64F);
         jd->alpha = alpha;
      }
      CUDA_CALL(double2halfMat(A->devValuesH, A->nnz, vec, A->nnz, A->nnz, 1));
      cudaFree(vec);
   }else if(jd->useHalf == -1){

      /* float precision matrix creation */
      double *vec; cudaMalloc((void**)&vec,(A->nnz)*sizeof(double));
      cudaMemcpy(vec,A->devValuesD,(A->nnz)*sizeof(double),cudaMemcpyDeviceToDevice);
      double alpha; 

      if(jd->normMatrix > 5e+07 || jd->normMatrix < 5e-07){
         alpha = 1e+05/(jd->normMatrix);
         cublasScalEx(*cublasH,A->nnz,&alpha,CUDA_R_64F,vec,CUDA_R_64F,1,CUDA_R_64F);
         jd->alpha = alpha;
      }
      CUDA_CALL(double2floatMat(A->devValuesF, A->nnz, vec, A->nnz, A->nnz, 1));
      cudaFree(vec);
   }

   return;
}

void destroy_jdqmr16(struct jdqmr16Info *jd){

   /* destroy Global Memory that is used in inner functions */
   cudaFree(jd->gpuMemSpaceDouble);
   cudaFree(jd->gpuMemSpaceInt);
   cudaFree(jd->gpuMemSpaceVoid);

   /* destroy inner functions */
   lock_destroy(jd);
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
   float  *devValsF = A->devValuesF;
   int    *devRows  = A->devRows;
   int    *devCols  = A->devCols;
   int    nnz       = A->nnz;
   int    dim       = A->dim;

   CUDA_CALL(cudaFree(devVals));
   CUDA_CALL(cudaFree(devValsH));
   CUDA_CALL(cudaFree(devValsF));
   CUDA_CALL(cudaFree(devCols));
   CUDA_CALL(cudaFree(devRows));


   struct devSolverSpace *sp = jd->sp;

   free(sp->normr);

   CUDA_CALL(cudaFree(sp->Qlocked));
   CUDA_CALL(cudaFree(sp->Llocked)); 

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
   
   double *Qlocked   = sp->Qlocked; int ldQlocked = sp->ldQlocked;
   double *Llocked   = sp->Llocked;
   int    &numLocked = sp->numLocked;

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
   double  maxerr;
   int     iter      = 0;

   jd->numMatVecsfp64 = 0;
   jd->numMatVecsfp16 = 0;

   double *normr =  sp->normr;//(double*)malloc(sizeof(double)*numEvals);
   // Step 0.1: Initialize matrices and basis
   initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,maxBasis,numEvals,1,jd); // basis initilization and H creation

   // Step 0.2: First approximation of eigenpairs
   eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors
   // Step 0.3: Residual calculation
   residual(R, ldR, V, ldV, L, numEvals, jd); 
   
   /* main loop of JDQMR */
   for(int i=0;i<maxIter;i++){   
      /* Inner sQMR16 to be used here in the future */
      //cudaMemcpy(P,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      innerSolver(P,ldP,R,ldR,normr,V,ldV,L,numEvals,dim,tol*normA, jd);
      
      if(basisSize == maxBasis){
         /* no space left - Restart basis */
         restart(W, ldW, H, ldH, Vprev, ldVprev, NULL ,
                  V, ldV, L, AW, ldAW, &basisSize, maxBasis, numEvals, dim, jd);
      }

      /* Enrich basis with new vectors*/    
      expandBasis(W, ldW, H, ldH, P, ldP, Qlocked, ldQlocked, numLocked, AW, ldAW, basisSize, dim,numEvals, jd);
      //basisSize++;
      
      /* keep previous ritz vectors for restarting purposes*/
      cudaMemcpy(Vprev,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      /* Find new Ritz pairs */
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize,jd);  // first approximation of eigevectors

      /* Residual calculation */
      residual(R, ldR, V, ldV, L, numEvals, jd); 

      /* convergence check */
      int    numConverged = 0;
      //numLocked = 0;
      for(int j=0;j<numEvals;j++){
         cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
      }
      /* locking new converged eigenpairs */
      lock(V,ldV,L,R,ldR,normr,Qlocked,ldQlocked,Llocked,W,ldW,H,ldH,AW,ldAW,
            numLocked,numEvals,maxBasis,basisSize,dim,tol*normA,jd);

      if(numLocked == numEvals){
         /* RR projection with new eigenpairs and break */
         cudaMemcpy(V,Qlocked,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
         initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,1,numEvals,0,jd); // basis initilization and H creation
         eigH(V, ldV, L, W,ldW, H, ldH, numEvals, 1,jd);  // first approximation of eigevectors
         residual(R, ldR, V, ldV, L, numEvals, jd); 
         int notFound = 0;
         for(int j=0;j<numEvals;j++){
            cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
         }

         break;
      }

      #if 0
      if(i%50 == 0){
         for(int j=0;j<numEvals;j++){
            printf("%%normr[%d]/normA = %e\n",i,normr[j]/normA);        
         }
         printf("%%-----\n");
      }
      #endif
      if(numConverged == numEvals){
         break;
      }
         

      jd->outerIterations++;
         
   }

}



void jdqmr16_eigenpairs(double *V, int ldV, double *L, double *normr, struct jdqmr16Info *jd){
   
   struct devSolverSpace* sp = jd->sp;

   cudaMemcpy(V,sp->V,sizeof(double)*(jd->numEvals)*ldV,cudaMemcpyDeviceToHost);
   cudaMemcpy(L,sp->L,sizeof(double)*(jd->numEvals),cudaMemcpyDeviceToHost);
   memcpy(normr,sp->normr,sizeof(double)*(jd->numEvals));

}































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
   int    *devCsrRows  = A->devCsrRows;

   int    *devCols  = A->devCols;
   int     nnz      = A->nnz;
   int     dim      = A->dim;
   int     numEvals = jd->numEvals;
   int     maxBasis = jd->maxBasis;

   jd->outerIterations = 0;
   jd->innerIterations = 0;

   if(jd->useHalf !=-1 && jd->useHalf != 0 && jd->useHalf!=1 && jd->useHalf!=-2 && jd->useHalf!=-3){
      jd->useHalf = 1;
   }
   /* if matrix is small */
   if(numEvals*maxBasis >= dim){
      jd->maxBasis = floor(dim/numEvals);
      maxBasis = jd->maxBasis;
   }
 
   /* initialize data to device */
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesD),sizeof(double)*nnz));
#if 0
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesF),sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devValuesH),sizeof(half)*nnz));
#endif

   CUDA_CALL(cudaMalloc((void**)&(A->devCols),sizeof(int)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devRows),sizeof(int)*nnz));
   CUDA_CALL(cudaMalloc((void**)&(A->devCsrRows),sizeof(int)*(dim+1)));

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

   // create csr 
   cusparseXcoo2csr(*cusparseH,A->devRows,nnz,dim,A->devCsrRows,CUSPARSE_INDEX_BASE_ZERO);


   /* initialize space for solver */
   struct devSolverSpace* sp = jd->sp;
   CUDA_CALL(cudaMalloc((void**)&sp->W,sizeof(double)*dim*maxBasis*numEvals));               sp->ldW     = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->H,sizeof(double)*maxBasis*numEvals*maxBasis*numEvals)); sp->ldH     = maxBasis*numEvals;
   CUDA_CALL(cudaMalloc((void**)&sp->Vprev,sizeof(double)*numEvals*dim));                    sp->ldVprev = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->V,sizeof(double)*numEvals*dim));                        sp->ldV     = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->L,sizeof(double)*numEvals)); 
   CUDA_CALL(cudaMalloc((void**)&sp->R,sizeof(double)*numEvals*dim));                        sp->ldR     = dim;
   CUDA_CALL(cudaMalloc((void**)&sp->QH,sizeof(double)*maxBasis*numEvals*maxBasis*numEvals));sp->ldQH    = maxBasis*numEvals;


//   CUDA_CALL(cudaMalloc((void**)&sp->Qlocked,sizeof(double)*numEvals*dim));                  sp->ldQlocked= dim;
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

   // init locking /* GPU Global Mem Done */
   jd->spLock = (struct lockSpace*)malloc(sizeof(struct lockSpace));
   sp->maxLockedVals = 2*numEvals;
/*   
   lock_init(sp->V, sp->ldV, sp->L, sp->R, sp->ldR, NULL,sp->Qlocked, sp->ldQlocked,
             sp->Llocked,sp->W,sp->ldW,sp->H,sp->ldH,sp->AW,sp->ldAW, 
            sp->numLocked, numEvals, maxBasis, sp->numLocked, dim, 1e-08,jd);
*/
   if(jd->locking != 1 && jd->locking != 0 ){
      jd->locking = 0;   
   }


   /* Allocate Global GPU Memory to be used by all inner functions */
   CUDA_CALL(cudaMalloc((void**)&(jd->gpuMemSpaceDouble),(jd->gpuMemSpaceDoubleSize)*sizeof(double)));
   CUDA_CALL(cudaMalloc((void**)&(jd->gpuMemSpaceInt),(jd->gpuMemSpaceIntSize)*sizeof(int)));
   CUDA_CALL(cudaMalloc((void**)&(jd->gpuMemSpaceVoid),(jd->gpuMemSpaceVoidSize)));

   /* find norm of matrix */
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

   /* destroy Global Memory that is used in inner functions */
   cudaFree(jd->gpuMemSpaceDouble);
   cudaFree(jd->gpuMemSpaceInt);
   cudaFree(jd->gpuMemSpaceVoid);

   /* destroy inner functions */
   //lock_destroy(jd);
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
   
   double *devVals    = A->devValuesD;
   half   *devValsH   = A->devValuesH;
   float  *devValsF   = A->devValuesF;
   int    *devRows    = A->devRows;
   int    *devCsrRows = A->devCsrRows;
   int    *devCols    = A->devCols;
   int    nnz       = A->nnz;
   int    dim       = A->dim;

   cudaFree(A->devValuesD);

#if 0
   cudaFree(A->devValuesH);
   cudaFree(A->devValuesF);
#endif
   cudaFree(A->devRows);
   cudaFree(A->devCsrRows);
   cudaFree(A->devCols);

   struct devSolverSpace *sp = jd->sp;

   free(sp->normr);

//   cudaFree(sp->Qlocked);
   cudaFree(sp->Llocked); 

   cudaFree(sp->AW);
   cudaFree(sp->P);
   cudaFree(sp->R);
   cudaFree(sp->QH);

   cudaFree(sp->W);
   cudaFree(sp->H);
   cudaFree(sp->Vprev);
   cudaFree(sp->V);
   cudaFree(sp->L);

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
   
//   double *Qlocked   = sp->Qlocked; int ldQlocked = sp->ldQlocked;
   double *Llocked   = sp->Llocked;
   int    &numLocked = sp->numLocked;

   double *R  = sp->R;  int ldR  = sp->ldR; /* Ritz vectors */
   double *QH = sp->QH; int ldQH = sp->ldQH; /* Eigenvectors of the projected system */

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
   eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize, QH, ldQH,jd);  // first approximation of eigevectors
   // Step 0.3: Residual calculation
   residual(R, ldR, V, ldV, L, AW, ldAW, QH, ldQH, numEvals,basisSize, jd);
   for(int j=0;j<numEvals;j++){
      cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
   }
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
      expandBasis(W, ldW, H, ldH, P, ldP, numLocked, AW, ldAW, basisSize, dim,numEvals, jd);
      //basisSize++;
      
      /* keep previous ritz vectors for restarting purposes*/
      cudaMemcpy(Vprev,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      /* Find new Ritz pairs */
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize, QH, ldQH, jd);  // first approximation of eigevectors

      /* Residual calculation */
      residual(R, ldR, W, ldW, L, AW, ldAW, QH, ldQH, numEvals,basisSize, jd);

      /* convergence check */
      int    numConverged = 0;
      //numLocked = 0;
      numLocked = 0;
      for(int j=0;j<numEvals;j++){
         cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
         if(normr[j] < tol*normA) numLocked++;
      }

      /* locking new converged eigenpairs */
      if(numLocked == numEvals){
         /* RR projection with new eigenpairs and break */
         initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,1,numEvals,0,jd); // basis initilization and H creation
         eigH(V, ldV, L, W,ldW, H, ldH, numEvals, 1, QH, ldQH, jd);  // first approximation of eigevectors
         residual(R, ldR, W, ldW, L, AW, ldAW, QH, ldQH, numEvals,basisSize, jd);

         int notFound = 0;
         for(int j=0;j<numEvals;j++){
            cublasDnrm2(jd->gpuH->cublasH,dim,&R[0+j*ldR], 1, &normr[j]);
         }

         break;
      }

      #if 0
      if(i%100 == 0){
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































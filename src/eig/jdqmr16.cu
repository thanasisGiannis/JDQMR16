#include "../../include/jdqmr16.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "../include/helper.h"
#include "initBasis.h"

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
   CUDA_CALL(cudaMalloc((void**)&sp->L,sizeof(double)*numEvals)); 
   
   double *H        = sp->H;        /* projected Matrix */
   double *V        = sp->V;        /* Ritz vectors */
   double *W        = sp->L;        /* Ritz values */



   /* init inner functions */
   jd->spInitBasis = (struct initBasisSpace *)malloc(sizeof(struct initBasisSpace));
   initBasis_init(sp->W,sp->ldW, sp->H, sp->ldH, sp->V, sp->ldV,sp->L, dim, maxBasis,numEvals,jd);


   return;
}

void destroy_jdqmr16(struct jdqmr16Info *jd){

   /* destroy inner functions */
   initBasis_destroy(jd);


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
   CUDA_CALL(cudaFree(sp->W));
   CUDA_CALL(cudaFree(sp->H));
   CUDA_CALL(cudaFree(sp->Vprev));
   CUDA_CALL(cudaFree(sp->V));
   CUDA_CALL(cudaFree(sp->L));

   free(jd->gpuH);
   free(jd->sp);
   
   return;

}

void jdqmr16(struct jdqmr16Info *jd){

   /* Generalized Davidson Iteration */
   struct jdqmr16Matrix *A = jd->matrix;   

   struct devSolverSpace* sp = jd->sp;
   double *W = sp->W; int ldW = sp->ldW; /* GD basis */
   double *H = sp->H; int ldH = sp->ldH; /* projected Matrix */
   double *V = sp->V; int ldV = sp->ldV; /* Ritz vectors */
   double *L = sp->L;                    /* Ritz values */

   int     dim      = A->dim;       /* dimension of the problem */
   int     numEvals = jd->numEvals; /* number of wanted eigenvalues */
   int     maxBasis = jd->maxBasis; /* maximum size of GD */

   int     basisSize = 1; // basis size in blocks 
   initBasis(W,ldW,H,ldH,V,ldV,L,dim,maxBasis,numEvals,jd);

}

































#include "../../include/jdqmr16.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper.h"

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
   CUDA_CALL(cudaMalloc((void**)&devVals,sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&devCols,sizeof(double)*nnz));
   CUDA_CALL(cudaMalloc((void**)&devRows,sizeof(double)*(nnz+1)));

   CUDA_CALL(cudaMemcpy((void*)devVals,(void*)vals,sizeof(double)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)devCols,(void*)cols,sizeof(double)*nnz,cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMemcpy((void*)devRows,(void*)rows,sizeof(double)*(nnz+1),cudaMemcpyHostToDevice));



   /* allocate device memory for solver */
   jd->sp = (struct devSolverSpace*)malloc(sizeof(struct devSolverSpace));
   struct devSolverSpace* sp = jd->sp;
   CUDA_CALL(cudaMalloc((void**)&sp->W,sizeof(double)*dim*maxBasis*numEvals));


   return;
}

void destroy_jdqmr16(struct jdqmr16Info *jd){
   /* allocate gpu memory */
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

   free(jd->sp);


   return;

}

void jdqmr16(){

   printf("jdqmr16\n");
}


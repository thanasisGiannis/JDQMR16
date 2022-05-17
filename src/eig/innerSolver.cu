
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

void applyDiagMat2Vec(double *D, double *x, int dim);
void updateScaledMatrix();


void innerSolver_init(double *X, int ldX, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

   /*
      calculate the column rows
      send them to device
      memory allocation    
   */

   
   struct jdqmr16Matrix  *matrix = jd->matrix;
   double  *A    = matrix->values;
	int     *cols = matrix->cols;
	int     *rows = matrix->rows;
   int      nnz  = matrix->nnz;
   /*
      Symmetric Matrix A 
   */
   double *norms  = (double*)malloc(sizeof(double)*dim);
   memset((void*)norms, 0,sizeof(double)*dim);
   
   for(int i=0; i<nnz; i++){
         norms[rows[i]] += A[i]*A[i];
   }

   for(int i=0; i<dim; i++){
      norms[i] = sqrt(norms[i]);
   }

   /*
      allocate memory for scaled matrix
   */

   cudaMalloc((void**)&(spInnerSolver->devScaledValuesD),sizeof(double)*nnz);
   cudaMalloc((void**)&(spInnerSolver->devRows),sizeof(int)*nnz);
   cudaMalloc((void**)&(spInnerSolver->devCols),sizeof(int)*nnz);

   cudaMalloc((void**)&(spInnerSolver->devNorms),sizeof(double)*dim);
   cudaMalloc((void**)&(spInnerSolver->devInvNorms),sizeof(double)*dim);


   /* 
      
   */
   cudaMemcpy(spInnerSolver->devRows,rows,sizeof(int)*nnz,cudaMemcpyHostToDevice);
   cudaMemcpy(spInnerSolver->devCols,cols,sizeof(int)*nnz,cudaMemcpyHostToDevice);
   cudaMemcpy(spInnerSolver->devScaledValuesD,A,sizeof(double)*nnz,cudaMemcpyHostToDevice);

   cudaMemcpy(spInnerSolver->devNorms,norms,sizeof(double)*dim,cudaMemcpyHostToDevice);

   for(int i=0; i<dim; i++){
      norms[i] = 1.0/norms[i];
   }

   cudaMemcpy(spInnerSolver->devInvNorms,norms,sizeof(double)*dim,cudaMemcpyHostToDevice);

   free(norms);
}


__global__
void applyDiagMat2Vec_Kernel(double *D, double *x,int dim){

   int i = blockIdx.x*blockDim.x+threadIdx.x;

   if(D == NULL || x == NULL) return;
   
   if(i<dim){
      x[i] = x[i]*D[i];
   }

}


void applyDiagMat2Vec(double *D, double *x, int dim){

   applyDiagMat2Vec_Kernel<<<ceil(dim/256),1>>>(D,x,dim);
}


void innerSolver_destroy(struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;
   
   cudaFree((spInnerSolver->devScaledValuesD));
   cudaFree((spInnerSolver->devRows));
   cudaFree((spInnerSolver->devCols));

   cudaFree((spInnerSolver->devNorms));
   cudaFree((spInnerSolver->devInvNorms));

}

void innerSolver(double *X, int ldX, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd){

   struct innerSolverSpace *spInnerSolver = jd->spInnerSolver;

/*
   double *A = devScaledValuesD; //double precision
   void   *A = devScaledValuesH; //half   precision
   int    *devRows;
   int    *devCols;
*/
   double *norms    = (double*)spInnerSolver->devNorms;
   double *invNorms = (double*)spInnerSolver->devInvNorms;


   cudaMemcpy(X,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
   
   
   /*
      first thing first to calculate the new scaled matrix
      taking account the new ritz values

      if needed cast vectors to reduced precision

      and then solve
   */
   for(int i=0;i<numEvals;i++){
      applyDiagMat2Vec(norms, &X[0+i*ldX], dim);
      applyDiagMat2Vec(invNorms, &X[0+i*ldX], dim);

   }
}

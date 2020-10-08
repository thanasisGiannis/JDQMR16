#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>

#include "residual.h"

#include "../include/helper.h"
#include "../../include/jdqmr16.h"



void residual_init(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd){
 
   struct jdqmr16Matrix  *A = jd->matrix;
   int dim    = A->dim;

   /* handlers for gpu and jdqmr16 */
   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;
   struct residualSpace *spRes = jd->spResidual;
   
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cublasHandle_t     cublasH   = gpuH->cublasH;
   cusparseHandle_t   cusparseH = gpuH->cusparseH;



   cudaMalloc((void**)&(spRes->VL),sizeof(double)*dim*numEvals);
   spRes->ldVL = dim;
   spRes->hL = (double*)malloc(sizeof(double)*numEvals);


   // R = AV
   cusparseCreateCoo(&(spRes->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesD,
							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
	cusparseCreateDnMat(&(spRes->descrV),dim,numEvals,ldV,(void*)V,CUDA_R_64F,CUSPARSE_ORDER_COL);
	cusparseCreateDnMat(&(spRes->descrR),dim,numEvals,ldR,(void*)R,CUDA_R_64F,CUSPARSE_ORDER_COL);

   double one  = 1.0;
   double zero = 0.0;

   cusparseSpMM_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spRes->descrA,spRes->descrV,&zero,
                        spRes->descrR,CUDA_R_64F,CUSPARSE_COOMM_ALG2,&(spRes->bufferSize));


   assert(spRes->bufferSize>0);
	cudaMalloc((void**)&(spRes->buffer),spRes->bufferSize);

}

void residual_destroy(struct jdqmr16Info *jd){

   struct residualSpace *spRes = jd->spResidual;
   
   cudaFree(spRes->buffer);
   cudaFree(spRes->VL);
   free(spRes->hL);


}


void residual(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd){


   
   struct jdqmr16Matrix  *A = jd->matrix;
   int dim    = A->dim;

   /* handlers for gpu and jdqmr16 */
   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;
   struct residualSpace *spRes = jd->spResidual;
   
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cublasHandle_t     cublasH   = gpuH->cublasH;
   cusparseHandle_t   cusparseH = gpuH->cusparseH;
   double *VL = spRes->VL; int ldVL = spRes->ldVL;
   
   
   // buffer matrices allocation mem
   // AV,VL, hL
   double *hL = spRes->hL;
   cudaMemcpy(spRes->hL,L,sizeof(double)*numEvals,cudaMemcpyDeviceToHost); 

   // VL = V
   cudaMemcpy(spRes->VL,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice); 
   // VL = VL*L
   double *alpha,*x;
	int incx = 1;

   cublasDdgmm(cublasH,CUBLAS_SIDE_RIGHT,dim,numEvals,VL,ldVL,L,1,VL,ldVL);
/*
   for(int i=0;i<numEvals;i++){
		
		alpha = &hL[i];
		x 	   = &(spRes->VL[0+spRes->ldVL*i]);
		
		cublasDscal(cublasH,dim,alpha,x,incx);
	}
*/
   // R = AV

   double one  = 1.0;
   double zero = 0.0;

   cusparseSpMM(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,spRes->descrA,spRes->descrV,&zero,spRes->descrR,CUDA_R_64F,CUSPARSE_COOMM_ALG2,spRes->buffer);


   /* R = R-VL */
   double minus_one = -1.0;
   cublasDgeam(cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim,numEvals,&one,R, ldR,
                          &minus_one,spRes->VL,spRes->ldVL,R,ldR);

}

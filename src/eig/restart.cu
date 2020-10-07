#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>

#include "restart.h"


#include "../include/helper.h"
#include "../../include/jdqmr16.h"

void restart_init(double *W, int ldW, double *H, int ldH, 
               double *Vprev, int ldVprev, double *Lprev,
               double *V, int ldV, double *L,
               int *basisSize, int maxBasisSize, int numEvals, int dim, 
               struct jdqmr16Info *jd){


   struct gpuHandler *gpuH    = jd->gpuH;
   cublasHandle_t     cublasH   = gpuH->cublasH;
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cusparseHandle_t   cusparseH = gpuH->cusparseH;

   struct restartSpace *spRestart = jd->spRestart;

   cudaMalloc((void**)&(spRestart->VprevTV),sizeof(double)*numEvals*numEvals);
   spRestart->ldVprevTV = numEvals;



   double *d_tau;  
   int    *devInfo;
   double *d_work;

   int lwork_geqrf = 0;
   int lwork_orgqr = 0;
   int lwork = 0;
   int info_gpu = 0;   

   cudaMalloc((void**)&(spRestart->d_tau), sizeof(double)*dim);
   cudaMalloc((void**)&(spRestart->devInfo), sizeof(int));

   
   cusolverDnDgeqrf_bufferSize(cusolverH,dim,numEvals,V,ldV,&lwork_geqrf);
   cusolverDnDorgqr_bufferSize(cusolverH,dim,numEvals,numEvals,V,ldV,spRestart->d_tau,&lwork_orgqr);

   spRestart->lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
   cudaMalloc((void**)&(spRestart->d_work), sizeof(double)*(spRestart->lwork));



   cusparseSpMatDescr_t descrA;
   cusparseDnMatDescr_t descrW;
   cusparseDnMatDescr_t descrAW;
   cudaMalloc((void**)&(spRestart->AW),sizeof(double)*dim*2*numEvals);
   spRestart->ldAW = dim;

   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseCreateCoo(&(spRestart->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesD,
							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
	cusparseCreateDnMat(&(spRestart->descrW),dim,2*numEvals,ldW,W,CUDA_R_64F,CUSPARSE_ORDER_COL);
	cusparseCreateDnMat(&(spRestart->descrAW),dim,2*numEvals,spRestart->ldAW,spRestart->AW,CUDA_R_64F,CUSPARSE_ORDER_COL);

   size_t bufferSize;
   double zero = 0.0;
   double one  = 1.0;
   cusparseSpMM_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spRestart->descrA,spRestart->descrW,&zero,
                        spRestart->descrAW,CUDA_R_64F,CUSPARSE_COOMM_ALG2,&(spRestart->bufferSize));

   assert(spRestart->bufferSize>0);
	cudaMalloc((void**)&(spRestart->buffer),spRestart->bufferSize);

}

void restart_destroy(struct jdqmr16Info *jd){

   struct restartSpace *spRestart = jd->spRestart;

   cudaFree(spRestart->AW);
   cudaFree(spRestart->VprevTV);

   cudaFree(spRestart->d_tau);
   cudaFree(spRestart->devInfo);
   cudaFree(spRestart->d_work);

   cudaFree(spRestart->buffer);
}

void restart(double *W, int ldW, double *H, int ldH, 
               double *Vprev, int ldVprev, double *Lprev,
               double *V, int ldV, double *L,
               int *basisSize, int maxBasisSize, int numEvals, int dim, 
               struct jdqmr16Info *jd){

   *basisSize = 2;

   struct gpuHandler *gpuH    = jd->gpuH;
   cublasHandle_t     cublasH   = gpuH->cublasH;
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cusparseHandle_t   cusparseH = gpuH->cusparseH;

   struct restartSpace *spRestart = jd->spRestart;

   cudaMemset(W,0,sizeof(double)*ldW*maxBasisSize*numEvals);
   cudaMemset(H,0,sizeof(double)*ldH*maxBasisSize*numEvals);

   /* W = [Vprev V]*/
   cudaMemcpy(W,Vprev,sizeof(double)*ldVprev*numEvals,cudaMemcpyDeviceToDevice);
   double *VprevTV   = spRestart->VprevTV;
   int     ldVprevTV = spRestart->ldVprevTV;


   double zero = 0.0;
   double one  = 1.0;

   for(int i=0; i<numEvals; i++){
      cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                              Vprev,CUDA_R_64F,ldVprev,V,CUDA_R_64F,ldV,&zero,
                              VprevTV,CUDA_R_64F,ldVprevTV,CUDA_R_64F,
                              CUBLAS_GEMM_ALGO2);
      double minus_one = -1.0;
      cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numEvals,&minus_one,
                              Vprev,CUDA_R_64F,ldVprev,VprevTV,CUDA_R_64F,ldVprevTV,&one,
                              V,CUDA_R_64F,ldV,CUDA_R_64F,
                              CUBLAS_GEMM_ALGO2);
   }

   double *d_tau = spRestart->d_tau;  
   int    *devInfo = spRestart->devInfo;
   double *d_work = spRestart->d_work;

   int lwork_geqrf = 0;
   int lwork_orgqr = 0;
   int lwork = spRestart->lwork;
   int info_gpu = 0;   


   cusolverDnDgeqrf(cusolverH,dim,numEvals,V,ldV,d_tau,d_work,lwork,devInfo);
   cusolverDnDorgqr(cusolverH,dim,numEvals,numEvals,V,ldV,d_tau,d_work,lwork,devInfo);

   cudaMemcpy(&W[0+numEvals*ldW],V,sizeof(double)*ldV*numEvals,cudaMemcpyDeviceToDevice);


   /* H = W'*A*W */
   cusparseSpMatDescr_t descrA = spRestart->descrA;
   cusparseDnMatDescr_t descrW = spRestart->descrW;
   cusparseDnMatDescr_t descrAW = spRestart->descrAW;

   double *AW = spRestart->AW;
   int ldAW = spRestart->ldAW;

   double *buffer = spRestart->buffer;

   cusparseSpMM(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,descrA,descrW,&zero,descrAW,CUDA_R_64F,
             CUSPARSE_COOMM_ALG2,buffer);

   cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,2*numEvals,2*numEvals,dim,&one,
                        W,CUDA_R_64F,ldW,AW,CUDA_R_64F,ldAW,&zero,
                        H,CUDA_R_64F,ldH,CUDA_R_64F,
                        CUBLAS_GEMM_ALGO2);

//   cublasDaxpy(cublasH,numEvals,&one,Lprev,1,H,ldH+1);


}

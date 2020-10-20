#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>

#include "expandBasis.h"

#include "../include/helper.h"
#include "../../include/jdqmr16.h"

void expandBasis_init(double *W, int ldW, double *H, int ldH, double *P, int ldP,
                int maxBasisSize, int dim, int numEvals, struct jdqmr16Info *jd){

   struct expandBasisSpace *spExpandBasis = jd->spExpandBasis;


   cudaMalloc((void**)&(spExpandBasis->AP),sizeof(double)*dim*numEvals);
   spExpandBasis->ldAP = dim;
   
   /* P = P - W*W'*P */
   struct gpuHandler     *gpuH      = jd->gpuH;
   cublasHandle_t         cublasH   = gpuH->cublasH;
   cusolverDnHandle_t     cusolverH = gpuH->cusolverH;
   cusparseHandle_t       cusparseH = gpuH->cusparseH;


   cudaMalloc((void**)&(spExpandBasis->WTP),sizeof(double)*maxBasisSize*numEvals*numEvals);
   spExpandBasis->ldWTP = maxBasisSize*numEvals;

   /* P = orth(P) */
   double *d_tau   = spExpandBasis->d_tau;
   int    *devInfo = spExpandBasis->devInfo;
   double *d_work  = spExpandBasis->d_work;

   int lwork_geqrf = 0;
   int lwork_orgqr = 0;
   int lwork = spExpandBasis->lwork;
   int info_gpu = 0;   

   cudaMalloc((void**)&(spExpandBasis->d_tau), sizeof(double)*dim);
   cudaMalloc((void**)&(spExpandBasis->devInfo), sizeof(int));

   
   cusolverDnDgeqrf_bufferSize(cusolverH,dim,numEvals,P,ldP,&lwork_geqrf);
   cusolverDnDorgqr_bufferSize(cusolverH,dim,numEvals,numEvals,P,ldP,spExpandBasis->d_tau,&lwork_orgqr);

   spExpandBasis->lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
   cudaMalloc((void**)&(spExpandBasis->d_work), sizeof(double)*(spExpandBasis->lwork));



   /* AP = A*P */
   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseCreateCoo(&(spExpandBasis->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesD,
							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
	cusparseCreateDnMat(&(spExpandBasis->descrP),dim,numEvals,ldP,(void*)P,CUDA_R_64F,CUSPARSE_ORDER_COL);
	cusparseCreateDnMat(&(spExpandBasis->descrAP),dim,numEvals,spExpandBasis->ldAP,(void*)spExpandBasis->AP,CUDA_R_64F,CUSPARSE_ORDER_COL);

   double one  = 1.0;
   double zero = 0.0;

   cusparseSpMM_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spExpandBasis->descrA,spExpandBasis->descrP,&zero,
                        spExpandBasis->descrAP,CUDA_R_64F,CUSPARSE_COOMM_ALG2,&(spExpandBasis->bufferSize));


   assert(spExpandBasis->bufferSize>0);
	cudaMalloc((void**)&(spExpandBasis->buffer),spExpandBasis->bufferSize);


}

void expandBasis_destroy(struct jdqmr16Info *jd){

   struct expandBasisSpace  *spExpandBasis = jd->spExpandBasis;

   cudaFree(spExpandBasis->WTP);
   cudaFree(spExpandBasis->d_tau);
   cudaFree(spExpandBasis->devInfo);
   cudaFree(spExpandBasis->d_work);

   
   cudaFree(spExpandBasis->buffer);
   
   cudaFree(spExpandBasis->AP);

}

void expandBasis(double *W, int ldW, double *H, int ldH, double *P, int ldP, double *Qlocked, int ldQlocked, int numLocked,
                double *AW, int ldAW, int &basisSize, int dim, int numEvals, struct jdqmr16Info *jd){

   
   struct gpuHandler        *gpuH          = jd->gpuH;
   struct expandBasisSpace  *spExpandBasis = jd->spExpandBasis;

   cublasHandle_t         cublasH   = gpuH->cublasH;
   cusolverDnHandle_t     cusolverH = gpuH->cusolverH;
   cusparseHandle_t       cusparseH = gpuH->cusparseH;

   /* P = -W*W'*P + P */
   double *WTP = spExpandBasis->WTP; int ldWTP = spExpandBasis->ldWTP;
   
   double one  = 1.0;
   double zero = 0.0;

//   for(int i=0; i<basisSize; i++){
   for(int i=0; i<1; i++){
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,basisSize*numEvals,numEvals,dim,&one,
                              W,CUDA_R_64F,ldW,P,CUDA_R_64F,ldP,&zero,
                              WTP,CUDA_R_64F,ldWTP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      double minus_one = -1.0;
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,basisSize*numEvals,&minus_one,
                              W,CUDA_R_64F,ldW,WTP,CUDA_R_64F,ldWTP,&one,
                              P,CUDA_R_64F,ldP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
   }


   /* P = - Qlocked*Qlocked'*P + P  */
   for(int i=0; i<1; i++){
      // using W'P same buffer space
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numLocked,numEvals,dim,&one,
                              Qlocked,CUDA_R_64F,ldQlocked,P,CUDA_R_64F,ldP,&zero,
                              WTP,CUDA_R_64F,ldWTP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      double minus_one = -1.0;
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numLocked,&minus_one,
                              Qlocked,CUDA_R_64F,ldQlocked,WTP,CUDA_R_64F,ldWTP,&one,
                              P,CUDA_R_64F,ldP,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
   }

   /* P = orth(P)  */
   double *d_tau   = spExpandBasis->d_tau;
   int    *devInfo = spExpandBasis->devInfo;
   double *d_work  = spExpandBasis->d_work;
   int     lwork   = spExpandBasis->lwork;

   cusolverDnDgeqrf(cusolverH,dim,numEvals,P,ldP,d_tau,d_work,lwork,devInfo);
   cusolverDnDorgqr(cusolverH,dim,numEvals,numEvals,P,ldP,d_tau,d_work,lwork,devInfo);

   /* AP = A*P */
   double *AP = spExpandBasis->AP; int ldAP = spExpandBasis->ldAP;
   cusparseSpMM(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,spExpandBasis->descrA,spExpandBasis->descrP,&zero,spExpandBasis->descrAP,CUDA_R_64F,
             CUSPARSE_COOMM_ALG2,spExpandBasis->buffer);

   /* H = [H W'*AP; P'*AW P'*AP*/
   // P'*AP   
   CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals,dim,&one,
                           P,CUDA_R_64F,ldP,AP,CUDA_R_64F,ldAP,&zero,
                           &H[basisSize*numEvals+(basisSize*numEvals)*ldH],CUDA_R_64F,ldH,CUDA_R_64F,
                           CUBLAS_GEMM_DEFAULT));


   // W'*AP
   CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,basisSize*numEvals,numEvals,dim,&one,
                           W,CUDA_R_64F,ldW,AP,CUDA_R_64F,ldAP,&zero,
                           &H[0+(basisSize*numEvals)*ldH],CUDA_R_64F,ldH,CUDA_R_64F,
                           CUBLAS_GEMM_DEFAULT));


   // P'*AW
   CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,basisSize*numEvals,dim,&one,
                           P,CUDA_R_64F,ldP,AW,CUDA_R_64F,ldAW,&zero,
                           &H[basisSize*numEvals+0*ldH],CUDA_R_64F,ldH,CUDA_R_64F,
                           CUBLAS_GEMM_DEFAULT));


   /* AW = [AW AP] */
   cudaMemcpy(&AW[0 + basisSize*numEvals*ldAW],AP,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
   /* W = [W P] */
   cudaMemcpy(&W[0 +basisSize*numEvals*ldW],P,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);


   basisSize++;
}















#include <curand.h>
#include <cublas.h>
#include <stdio.h>
#include <assert.h>

#include "initBasis.h"


#include "../include/helper.h"
#include "../../include/jdqmr16.h"

void initBasis_init(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd){

   struct gpuHandler *gpuH = jd->gpuH;   
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   struct initBasisSpace *spInitBasis = jd->spInitBasis;

   curandGenerator_t curandH = gpuH->curandH;
	curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */

   int lwork_geqrf;
   int lwork_orgqr;
   int lwork      ;

   int info_gpu = 0;

   cudaMalloc ((void**)&(spInitBasis->d_tau), sizeof(double)*dim);
   cudaMalloc ((void**)&(spInitBasis->devInfo), sizeof(int));
   cudaMalloc ((void**)&(spInitBasis->d_R ) , sizeof(double)*numEvals*numEvals);

   cusolverDnDgeqrf_bufferSize(cusolverH,dim,numEvals,V,ldV,&lwork_geqrf);
   cusolverDnDorgqr_bufferSize(cusolverH,dim,numEvals,numEvals,V,ldV,spInitBasis->d_tau,&lwork_orgqr);
   spInitBasis->lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
   cudaMalloc((void**)&(spInitBasis->d_work), sizeof(double)*(spInitBasis->lwork));


   /* allocation of extra space */
   cudaMalloc((void**)&spInitBasis->AV,sizeof(double)*dim*numEvals); spInitBasis->ldAV = dim;// AV


   /* cusparse descriptors */
   double one  = 1.0;
   double zero = 0.0;
   cusparseHandle_t cusparseH = gpuH->cusparseH;
   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseCreateCoo(&(spInitBasis->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesD,
               							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);

	cusparseCreateDnMat(&(spInitBasis->descrV),dim,numEvals,dim,V,CUDA_R_64F,CUSPARSE_ORDER_COL);

   cusparseCreateDnMat(&(spInitBasis->descrAV),dim,numEvals,dim,spInitBasis->AV,CUDA_R_64F,CUSPARSE_ORDER_COL);
	cudaDeviceSynchronize();
   size_t bufferSize = -1;
	cusparseSpMM_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spInitBasis->descrA,spInitBasis->descrV,&zero,spInitBasis->descrAV,
                        CUDA_R_64F,CUSPARSE_COOMM_ALG2,&bufferSize);


   cudaMalloc((void**)&(spInitBasis->externalBuffer),bufferSize);
   spInitBasis->bufferSize = bufferSize;    

}


void initBasis_destroy(struct jdqmr16Info *jd){

   struct initBasisSpace *spInitBasis = jd->spInitBasis;

   cudaFree(spInitBasis->externalBuffer);
   cudaFree(spInitBasis->d_tau);
   cudaFree(spInitBasis->devInfo);
   cudaFree(spInitBasis->d_R);
   cudaFree(spInitBasis->d_work);
   cudaFree(spInitBasis->AV);
}

void initBasis(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L, double *AW, int ldAW,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd){


   struct gpuHandler *gpuH = jd->gpuH;   
   /* Step 1: Random initialization of V */
   curandGenerator_t curandH = gpuH->curandH;
	curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */

	double  mean = 0.0;
	double  stddev = max(dim,numEvals);

   cudaMemset(V,0,dim*numEvals);
	curandGenerateNormalDouble(curandH, V, dim*numEvals,mean,stddev); /* Generate dim*maxSizeW on device */

   /* Step 2: Orthogonalization of V */
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   struct initBasisSpace *spInitBasis = jd->spInitBasis;

   double *d_tau = spInitBasis->d_tau;
   int    *devInfo = spInitBasis->devInfo;
   double *d_work = spInitBasis->d_work;

   double *d_R = NULL;
   int info_gpu = 0;

   cusolverDnDgeqrf(cusolverH,dim,numEvals,V,ldV,d_tau,d_work,spInitBasis->lwork,devInfo);
   cusolverDnDorgqr(cusolverH,dim,numEvals,numEvals,V,ldV,d_tau,d_work,spInitBasis->lwork,devInfo);


   /* Projection of A into V : H = V'AV  */
   struct devSolverSpace *sp = jd->sp;
   cusparseHandle_t cusparseH = gpuH->cusparseH;
   cudaMemset(H,0,maxSizeW*maxSizeW*sizeof(double));
   double *AV = spInitBasis->AV; cudaMemset(AV,0,dim*numEvals*sizeof(double));
   int ldAV = spInitBasis->ldAV;

   struct jdqmr16Matrix  *A = jd->matrix;

	assert(spInitBasis->descrA != NULL || spInitBasis->descrV != NULL || spInitBasis->descrAV != NULL);
	cudaDeviceSynchronize();


   double one  = 1.0;
	double zero = 0.0;

	size_t bufferSize = -1;

   cusparseSpMM(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,
             spInitBasis->descrA,spInitBasis->descrV,&zero,spInitBasis->descrAV,
             CUDA_R_64F,CUSPARSE_COOMM_ALG2,spInitBasis->externalBuffer);


   cudaMemcpy(AW,AV,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

   cublasHandle_t cublasH = gpuH->cublasH;

   cublasDgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals, dim,&one,V,ldV,AV,ldAV,&zero,H,ldH);

   /* W = V */
   cudaMemcpy(W,V,dim*numEvals*sizeof(double),cudaMemcpyDeviceToDevice); 

}



















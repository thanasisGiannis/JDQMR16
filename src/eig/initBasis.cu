#include <curand.h>
#include <cublas.h>
#include <stdio.h>
#include <assert.h>

#include "initBasis.h"


#include "../include/helper.h"
#include "../../include/jdqmr16.h"

void initBasis_init(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd){

   int memReqD    = 0;
   int memReqI    = 0;
   size_t memReqV = 0;
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
//   cusparseCreateCoo(&(spInitBasis->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesD,
//               							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);

   cusparseCreateCsr(&(spInitBasis->descrA),dim,dim,A->nnz,A->devCsrRows,A->devCols,A->devValuesD,CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);


	cusparseCreateDnMat(&(spInitBasis->descrV),dim,numEvals,dim,V,CUDA_R_64F,CUSPARSE_ORDER_COL);
   cusparseCreateDnMat(&(spInitBasis->descrAV),dim,numEvals,dim,spInitBasis->AV,CUDA_R_64F,CUSPARSE_ORDER_COL);

	cudaDeviceSynchronize();
   size_t bufferSize = -1;
	cusparseSpMM_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spInitBasis->descrA,spInitBasis->descrV,&zero,spInitBasis->descrAV,
                        CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,&bufferSize);


   cudaMalloc((void**)&(spInitBasis->externalBuffer),bufferSize);
   spInitBasis->bufferSize = bufferSize;    

   memReqV += bufferSize;
   memReqD += dim*numEvals;
   memReqD += spInitBasis->lwork;
   memReqD += dim;
   memReqI += 1;

   jd->gpuMemSpaceDoubleSize = max(jd->gpuMemSpaceDoubleSize,memReqD);
   jd->gpuMemSpaceIntSize    = max(jd->gpuMemSpaceIntSize,memReqI);
   jd->gpuMemSpaceVoidSize   = max(jd->gpuMemSpaceVoidSize,memReqV);

#if 0
   cudaFree(spInitBasis->d_tau);
   cudaFree(spInitBasis->devInfo);
   cudaFree(spInitBasis->d_work);
   cudaFree(spInitBasis->externalBuffer);
   cudaFree(spInitBasis->AV);
#endif
}


void initBasis_destroy(struct jdqmr16Info *jd){

   struct initBasisSpace *spInitBasis = jd->spInitBasis;

#if 1
   cudaFree(spInitBasis->externalBuffer);
   cudaFree(spInitBasis->d_tau);
   cudaFree(spInitBasis->devInfo);
   cudaFree(spInitBasis->d_work);
   cudaFree(spInitBasis->AV);
#endif
}

void initBasis(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L, double *AW, int ldAW,
                int dim, int maxSizeW, int numEvals, int seed, struct jdqmr16Info *jd){


   struct gpuHandler *gpuH = jd->gpuH;   
   /* Step 1: Random initialization of V */

   if(seed == 1){
      curandGenerator_t curandH = gpuH->curandH;
	   curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */

	   double  mean = 0.0;
	   double  stddev = max(dim,numEvals);

      cudaMemset(V,0,dim*numEvals);
	   curandGenerateNormalDouble(curandH, V, dim*numEvals,mean,stddev); /* Generate dim*maxSizeW on device */
   }
   /* Step 2: Orthogonalization of V */
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   struct initBasisSpace *spInitBasis = jd->spInitBasis;

#if 0

   double *memD = jd->gpuMemSpaceDouble;
   int    *memI = jd->gpuMemSpaceInt;
   void   *memV = jd->gpuMemSpaceVoid;

   double *d_tau  = memD; memD+= dim;
   double *AV     = memD; memD+= dim;// spInitBasis->AV; 
   int ldAV = spInitBasis->ldAV;

   double *d_work = memD;
   void   *buffer = memV;
   int    *devInfo = memI;

   cusparseCreateDnMat(&(spInitBasis->descrAV),dim,numEvals,dim,AV,CUDA_R_64F,CUSPARSE_ORDER_COL);

#else

   double *d_tau  = spInitBasis->d_tau;
   double *d_work = spInitBasis->d_work;
   void   *buffer = spInitBasis->externalBuffer;
   int    *devInfo = spInitBasis->devInfo;
   double *AV = spInitBasis->AV; 
   int ldAV = spInitBasis->ldAV;
 
#endif



   int info_gpu = 0;

   cusolverDnDgeqrf(cusolverH,dim,numEvals,V,ldV,d_tau,d_work,spInitBasis->lwork,devInfo);
   cusolverDnDorgqr(cusolverH,dim,numEvals,numEvals,V,ldV,d_tau,d_work,spInitBasis->lwork,devInfo);


   /* Projection of A into V : H = V'AV  */
   struct devSolverSpace *sp = jd->sp;
   cusparseHandle_t cusparseH = gpuH->cusparseH;
   cudaMemset(H,0,maxSizeW*maxSizeW*sizeof(double));
   //double *AV = spInitBasis->AV; 
   cudaMemset(AV,0,dim*numEvals*sizeof(double));
  
   struct jdqmr16Matrix  *A = jd->matrix;

	assert(spInitBasis->descrA != NULL || spInitBasis->descrV != NULL || spInitBasis->descrAV != NULL);
	cudaDeviceSynchronize();


   double one  = 1.0;
	double zero = 0.0;

	size_t bufferSize = -1;

   cusparseSpMM(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,
             spInitBasis->descrA,spInitBasis->descrV,&zero,spInitBasis->descrAV,
             CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,buffer);
   jd->numMatVecsfp64++;

   cudaMemcpy(AW,AV,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

   cublasHandle_t cublasH = gpuH->cublasH;

   cublasDgemm(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numEvals,numEvals, dim,&one,V,ldV,AV,ldAV,&zero,H,ldH);

   /* W = V */
   cudaMemcpy(W,V,dim*numEvals*sizeof(double),cudaMemcpyDeviceToDevice); 

}



















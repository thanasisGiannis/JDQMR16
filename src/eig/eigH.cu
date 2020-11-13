#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>

#include "eigH.h"


#include "../include/helper.h"
#include "../../include/jdqmr16.h"

void eigH_init(double *W, int ldW, double *L, double *H, int ldH, int numEvals, int maxBasisSize, struct jdqmr16Info *jd){


   int memReqD    = 0;
   int memReqI    = 0;
   size_t memReqV = 0;


   int sizeQH = numEvals*maxBasisSize;

   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;

   int lwork      = spEig->lwork;   
   int *devInfo   = spEig->devInfo;
   cudaMalloc((void**)&(spEig->devInfo),sizeof(int));
   
   double *d_work = spEig->d_work;
   
   spEig->ldQH = numEvals*maxBasisSize;
   cudaMalloc((void**)&(spEig->QH), sizeof(double)*(spEig->ldQH)*(numEvals*maxBasisSize));

   cudaMalloc((void**)&(spEig->LH), sizeof(double)*(numEvals*maxBasisSize));

   cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

   cusolverDnDsyevd_bufferSize(cusolverH,jobz,uplo,sizeQH,spEig->QH,spEig->ldQH,spEig->LH,&(spEig->lwork));

   cudaMalloc((void**)&(spEig->d_work), sizeof(double)*(spEig->lwork));


   memReqD += spEig->lwork;
   memReqD += numEvals*maxBasisSize;
   memReqD += (spEig->ldQH)*(numEvals*maxBasisSize);
   memReqI+=1;

   jd->gpuMemSpaceDoubleSize = max(jd->gpuMemSpaceDoubleSize,memReqD);
   jd->gpuMemSpaceIntSize    = max(jd->gpuMemSpaceIntSize,memReqI);
   jd->gpuMemSpaceVoidSize   = max(jd->gpuMemSpaceVoidSize,memReqV);

   cudaFree(spEig->QH);
   cudaFree(spEig->LH);
   cudaFree(spEig->devInfo);
   cudaFree(spEig->d_work);

}

void eigH_destroy(struct jdqmr16Info *jd){

   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;

//   cudaFree(spEig->QH);
//   cudaFree(spEig->LH);
//   cudaFree(spEig->devInfo);
//   cudaFree(spEig->d_work);


}



void eigH(double *V, int ldV, double *L, double *W, int ldW, double *H, int ldH, int numEvals, int basisSize, struct jdqmr16Info *jd){




   int sizeQH = numEvals*basisSize;
   struct jdqmr16Matrix  *matrix = jd->matrix;
   int dim    = matrix->dim;
   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cublasHandle_t     cublasH   = gpuH->cublasH;

   /* copy H to QH so syevd can handle the eigenvectors correctly */ 
#if 1
   double *memD = jd->gpuMemSpaceDouble;
   int    *memI = jd->gpuMemSpaceInt;
   void   *memV = jd->gpuMemSpaceVoid;


   double *LH      = memD; memD += (numEvals*basisSize);
   int     ldQH    = spEig->ldQH;
   double *QH      = memD; memD += ldQH*numEvals*basisSize;
   double *d_work  = memD;
   int    *devInfo = memI;

#else
   double *LH   = spEig->LH;
   double *QH   = spEig->QH;
   int     ldQH = spEig->ldQH;
   double *d_work = spEig->d_work;
   int    *devInfo = spEig->devInfo;
#endif
   double one  = 1.0;
   double zero = 0.0;

   cudaMemset(QH,0,sizeof(double)*ldQH*sizeQH);
   cublasDgeam(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,sizeQH,sizeQH,&one,H, ldH,&zero,QH, ldQH, QH,ldQH);

   cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
   
   cusolverDnDsyevd(cusolverH,jobz,uplo,sizeQH,QH,ldQH,LH,d_work,spEig->lwork,devInfo);
   /* eigenvalues are in a ascending order */
   /* next step to choose which part of the spectrum is needed (smallest or largest) */
   /* for starters we get smallest */
   cudaMemcpy(L,LH,numEvals*sizeof(double),cudaMemcpyDeviceToDevice);

   /* V = W*QH */
   cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,basisSize*numEvals,&one,
                           W,CUDA_R_64F,ldW,QH,CUDA_R_64F,ldQH,&zero,
                           V,CUDA_R_64F,ldV,CUDA_R_64F,
                           CUBLAS_GEMM_ALGO2);

}




















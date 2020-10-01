#include <curand.h>
#include "initBasis.h"
#include <stdio.h>
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

}


void initBasis_destroy(struct jdqmr16Info *jd){

   struct initBasisSpace *spInitBasis = jd->spInitBasis;
   cudaFree(spInitBasis->d_tau);
   cudaFree(spInitBasis->devInfo);
   cudaFree(spInitBasis->d_R);
   cudaFree(spInitBasis->d_work);

}

void initBasis(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd){


   struct gpuHandler *gpuH = jd->gpuH;   
   /* Step 1: Random initialization of V */
   curandGenerator_t curandH = gpuH->curandH;
	curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */

	double  mean = 0.0;
	double  stddev = max(dim,numEvals);

   cudaMemset((void**)V,0,dim*numEvals);
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
   printMatrixDouble(V,dim,numEvals,"V");
}

#include <curand.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <assert.h>

#include "residual.h"

#include "../include/helper.h"
#include "../../include/jdqmr16.h"




void residual_init(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd){}
void residual_destroy(struct jdqmr16Info *jd){}


void residual(double *R, int ldR, double *V, int ldV, double *L, double *AV, int ldAV, double *QH, int ldQH,
               int numEvals, int basisSize, struct jdqmr16Info *jd){

//     R = AV*QH-V*QH*L;
   struct jdqmr16Matrix  *A = jd->matrix;
   int dim    = A->dim;

   /* handlers for gpu and jdqmr16 */
   struct gpuHandler *gpuH    = jd->gpuH;
   struct eigHSpace  *spEig   = jd->spEigH;
   struct residualSpace *spRes = jd->spResidual;
   
   cusolverDnHandle_t cusolverH = gpuH->cusolverH;
   cublasHandle_t     cublasH   = gpuH->cublasH;
   cusparseHandle_t   cusparseH = gpuH->cusparseH;

   double one  = 1.0;
   double zero = 0.0;
   double minus_one = -1.0;

   // R = AV*QH
   cublasDgemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim,numEvals, basisSize*numEvals,
                           &one,AV, ldAV,QH,ldQH,&zero,R,ldR);

   // QH = QH*L
   cublasDdgmm(cublasH,CUBLAS_SIDE_RIGHT,basisSize*numEvals,numEvals,QH,ldQH,L,1,QH,ldQH);

   // R = R-V*QH;
   cublasDgemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim,numEvals, basisSize*numEvals,
                           &minus_one,V, ldV,QH,ldQH,&one,R,ldR);
}








#include "../../include/jdqmr16.h"
#include "../include/helper.h"
#include "../matrix/double2halfMat.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "initBasis.h"
#include "eigH.h"
#include "residual.h"   
#include "locking.h"

void lock_init(double *V, int ldV, double *L, double *R, int ldR, double *normr,
            double *Qlocked, int ldQlocked, double *Llocked, double *W, int ldW, double *H, int ldH, double *AW, int ldAW, 
            int &numLocked, int &numEvals, int maxBasis, int &basisSize, int dim, double tol, struct jdqmr16Info *jd){

   int memReqD    = 0;
   int memReqI    = 0;
   size_t memReqV = 0;

   struct lockSpace *spLock = jd->spLock;

   cudaMalloc((void**)&(spLock->QTV),sizeof(double)*numEvals*numEvals);
   spLock->ldQTV = numEvals;

   cudaMalloc((void**)&(spLock->QTR),sizeof(double)*numEvals*numEvals);
   spLock->ldQTR = numEvals; 

   cudaMalloc((void**)&(spLock->PR),sizeof(double)*dim*numEvals);
   spLock->ldPR = dim;

   spLock->Lh = (double *)malloc(sizeof(double)*numEvals);
   spLock->Llockedh = (double *)malloc(sizeof(double)*numEvals);


   memReqD += numEvals*numEvals;
   memReqD += numEvals*numEvals;
   memReqD += dim*numEvals;



   jd->gpuMemSpaceDoubleSize = max(jd->gpuMemSpaceDoubleSize,memReqD);
   jd->gpuMemSpaceIntSize    = max(jd->gpuMemSpaceIntSize,memReqI);
   jd->gpuMemSpaceVoidSize   = max(jd->gpuMemSpaceVoidSize,memReqV);

//   return;
#if 0
   cudaFree(spLock->QTV);
   cudaFree(spLock->QTR);
   cudaFree(spLock->PR);
#endif
}

void lock_destroy(struct jdqmr16Info *jd){


   struct lockSpace *spLock = jd->spLock;

   free(spLock->Lh);
   free(spLock->Llockedh);
#if 1
   cudaFree(spLock->QTV);
   cudaFree(spLock->QTR);
   cudaFree(spLock->PR);
#endif
}


void lock(double *V, int ldV, double *L, double *R, int ldR, double *normr,
            double *Qlocked, int ldQlocked, double *Llocked, double *W, int ldW,
            double *H, int ldH, double *AW, int ldAW, double *QH,
            int ldQH, int &numLocked, int &numEvals, int maxBasis, int &basisSize, 
            int dim, double tol, struct jdqmr16Info *jd){


   if(jd->locking == 0){
      numLocked = 0;
      for(int i=0; i<numEvals; i++){
         if(normr[i] < tol){
            numLocked ++;
         }
      }
      if(numLocked == numEvals){
         cudaMemcpy(Qlocked,V,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      }

      return;
   }
   struct lockSpace *spLock = jd->spLock;

#if 1

   double *memD = jd->gpuMemSpaceDouble;
   int    *memI = jd->gpuMemSpaceInt;
   void   *memV = jd->gpuMemSpaceVoid;

   double *QTV = memD; memD += numEvals*numEvals;
   double *QTR = memD; memD += numEvals*numEvals;
   double *PR  = memD; 

   int ldQTV =  spLock->ldQTV;// numEvals;
   int ldQTR = spLock->ldQTR; 
   int ldPR = spLock->ldPR;

#else

   double *QTV = spLock->QTV; //cudaMalloc((void**)&QTV,sizeof(double)*numEvals*numEvals);
   int ldQTV =  spLock->ldQTV;// numEvals;
   double *QTR = spLock->QTR;// cudaMalloc((void**)&QTR,sizeof(double)*numEvals*numEvals);
   int ldQTR = spLock->ldQTR; 
   double *PR = spLock->PR;// cudaMalloc((void**)&PR,sizeof(double)*dim*numEvals);
   int ldPR = spLock->ldPR;

#endif

   /* V = V - Q*Q'*V */
   struct gpuHandler *gpuH = jd->gpuH;   
   cublasHandle_t cublasH = gpuH->cublasH;

   int loopIters = numEvals;
   int numEvalsFound = 0;
   for(int i=0; i<numEvals; i++){
      if(normr[i] < tol){
         /* lock eigenvalues */
         cudaMemcpy(&Qlocked[0+numLocked*ldQlocked],&V[0+i*ldV],dim*sizeof(double),cudaMemcpyDeviceToDevice);
         cudaMemcpy(&Llocked[numLocked],&L[i],sizeof(double),cudaMemcpyDeviceToDevice);

         /* put random vector inside previous spot of eigenvalues */
         if(jd->locking == 1){
            curandGenerator_t curandH = gpuH->curandH;
            curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */
            double  mean = 0.0;
            double  stddev = max(dim,numEvals);

            curandGenerateNormalDouble(curandH,&V[0+i*ldV],dim,mean,stddev);

            if(numLocked >= numEvals){
               numLocked = numEvals;
               break;
            }
 
         }
         numEvalsFound++;
         numLocked++;
      }
   }

   if(numLocked >= numEvals){
      numLocked = numEvals;
      return;
   }


   if(numEvalsFound>0){
      double minus_one = -1.0;
      double zero      =  0.0;
      double one       =  1.0;
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numLocked,numEvals,dim,&one,
                              Qlocked,CUDA_R_64F,ldQlocked,V,CUDA_R_64F,ldV,&zero,
                              QTV,CUDA_R_64F,ldQTV,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));

      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numLocked,&minus_one,
                              Qlocked,CUDA_R_64F,ldQlocked,QTV,CUDA_R_64F,ldQTV,&one,
                              V,CUDA_R_64F,ldV,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));

      /* init basis with new V */
      initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,maxBasis,numEvals,0,jd); 
      basisSize = 1;
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize, QH, ldQH, jd);
      //residual(R, ldR, V, ldV, L, numEvals, jd);
      residual(R, ldR, V, ldV, L, AW, ldAW, QH, ldQH, numEvals,basisSize, jd);
      return;
   }


   if(numLocked<1){
      return;
   }

   /* check residual with the complement space */
   double E = sqrt(numLocked)*tol;
   //printf("E=%e\n",E);
   /* PR = R-Q*Q*'*R */
   cudaMemcpy(PR,R,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

   double minus_one = -1.0;
   double zero      =  0.0;
   double one       =  1.0;

   CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numLocked,numEvals,dim,&one,
                           Qlocked,CUDA_R_64F,ldQlocked,PR,CUDA_R_64F,ldPR,&zero,
                           QTR,CUDA_R_64F,ldQTR,CUDA_R_64F,
                           CUBLAS_GEMM_DEFAULT));

   cudaDeviceSynchronize();
   CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numLocked,&minus_one,
                           Qlocked,CUDA_R_64F,ldQlocked,QTR,CUDA_R_64F,ldQTR,&one,
                           PR,CUDA_R_64F,ldPR,CUDA_R_64F,
                           CUBLAS_GEMM_DEFAULT));

   double *Lh = spLock->Lh; //(double *)malloc(sizeof(double)*numEvals);
   double *Llockedh = spLock->Llockedh;//(double *)malloc(sizeof(double)*numEvals);
   cudaMemcpy(Lh,L,sizeof(double)*numEvals,cudaMemcpyDeviceToHost);
   cudaMemcpy(Llockedh,Llocked,sizeof(double)*numLocked,cudaMemcpyDeviceToHost);

   for(int i=0;i<numEvals;i++){
      if(normr[i] < E){
         double normrd;
         cublasDnrm2(jd->gpuH->cublasH,dim,&PR[0+i*ldR], 1, &normrd);
         double vita = sqrt(normr[i]*normr[i]-normrd*normrd);
         E = sqrt(tol*tol+vita*vita);
         double gama,gamad,gamap;
         gamad = abs(Lh[i] - Llockedh[0]);
         int index = 0;
         for(int j=1;j<numLocked; j++){
            if(gamad > abs(Lh[i] - Llockedh[j])){            
               gamad = abs(Lh[i] - Llockedh[j]);
               index = j;
            }
         }

         gamap = 1e+500;
         for(int j=0; j<numLocked; j++){
            if(j == i) continue;
            if(gamap > abs(Lh[i] - Lh[j])){            
               gamap = abs(Lh[i] - Lh[j]);
               index = j;
            }
         }

         gama = min(gamap,gamad);

         if(vita > tol && normrd < tol*(gamap/gama)-(tol*tol*numLocked)/gamad){
            /* lock eigenvalues */
            printf("Hey HO!\n");
            cudaMemcpy(&Qlocked[0+numLocked*ldQlocked],&V[0+i*ldV],dim*sizeof(double),cudaMemcpyDeviceToDevice);
            cudaMemcpy(&Llocked[numLocked],&L[i],sizeof(double),cudaMemcpyDeviceToDevice);

            /* put random vector inside previous spot of eigenvalues */
            curandGenerator_t curandH = gpuH->curandH;
            curandSetPseudoRandomGeneratorSeed(curandH,1234ULL); /* set Seed */
            double  mean = 0.0;
            double  stddev = max(dim,numEvals);

//            cudaMemset(V,0,dim*numEvals);
            curandGenerateNormalDouble(curandH,&V[0+i*ldV],dim,mean,stddev); 
            numEvalsFound++;
         
            numLocked++;
            E = sqrt(numLocked)*tol;
            break;
         }
         
      }
   }

   if(numLocked >= numEvals){
      numLocked = numEvals;
      return;
   }

//return;

   if(numEvalsFound>0){

      /* V = V - Q*Q'*V */
      double minus_one = -1.0;
      double zero      =  0.0;
      double one       =  1.0;

      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,numLocked,numEvals,dim,&one,
                              Qlocked,CUDA_R_64F,ldQlocked,V,CUDA_R_64F,ldV,&zero,
                              QTV,CUDA_R_64F,ldQTV,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));
      CUBLAS_CALL(cublasGemmEx(cublasH,CUBLAS_OP_N,CUBLAS_OP_N,dim,numEvals,numLocked,&minus_one,
                              Qlocked,CUDA_R_64F,ldQlocked,QTV,CUDA_R_64F,ldQTV,&one,
                              V,CUDA_R_64F,ldV,CUDA_R_64F,
                              CUBLAS_GEMM_DEFAULT));

      /* init basis with new V */
      initBasis(W,ldW,H,ldH,V,ldV,L, AW, ldAW, dim,maxBasis,numEvals,0,jd); 
      basisSize = 1;
      eigH(V, ldV, L, W,ldW, H, ldH, numEvals, basisSize, QH, ldQH, jd);
      //residual(R, ldR, V, ldV, L, numEvals, jd);
      residual(R, ldR, V, ldV, L, AW, ldAW, QH, ldQH, numEvals,basisSize, jd);
      return;
   }
}

















#include "../../include/jdqmr16.h"
#include "../include/helper.h"
#include <curand.h>
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "sqmr.h"

#include <math.h>
//#include <tgmath.h>
__global__ void initIdentityGPU(double *devMatrix, int ldMat, int numR, int numC) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < numR && x < numC) {
          if(x == y)
              devMatrix[y + x*ldMat] = 1;
          else
              devMatrix[y + x*ldMat] = 0;
    }
}




void blQmrD_init(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){


   double one       =  1.0;
   double zero      =  0.0;
   double minus_one = -1.0;

   cudaMalloc((void**)&(spBlQmr->rin),sizeof(double)*dim*maxNumEvals);
   spBlQmr->ldrin=dim;
   cudaMalloc((void**)&(spBlQmr->w),sizeof(double)*dim*maxNumEvals);   
   spBlQmr->ldw  =dim;

   cudaMalloc((void**)&(spBlQmr->v1),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldv1=dim;   
   cudaMalloc((void**)&(spBlQmr->v2),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldv2=dim;      
   cudaMalloc((void**)&(spBlQmr->v3),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldv3=dim;      

   cudaMalloc((void**)&(spBlQmr->p0),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldp0=dim;     
   cudaMalloc((void**)&(spBlQmr->p1),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldp1=dim;     
   cudaMalloc((void**)&(spBlQmr->p2),sizeof(double)*dim*maxNumEvals); 
   spBlQmr->ldp2=dim;    

   cudaMalloc((void**)&(spBlQmr->qq),sizeof(double)*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldqq=2*maxNumEvals;    
   cudaMalloc((void**)&(spBlQmr->q0),sizeof(double)*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq0=2*maxNumEvals;       
   cudaMalloc((void**)&(spBlQmr->q1),sizeof(double)*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq1=2*maxNumEvals;       
   cudaMalloc((void**)&(spBlQmr->q2),sizeof(double)*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq2=2*maxNumEvals;       


   cudaMalloc((void**)&(spBlQmr->alpha),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldalpha=maxNumEvals;          
   cudaMalloc((void**)&(spBlQmr->vita2),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldvita2=maxNumEvals;             
   cudaMalloc((void**)&(spBlQmr->vita3),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldvita3=maxNumEvals;                
   cudaMalloc((void**)&(spBlQmr->tau2_),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldtau2_=maxNumEvals;                
   cudaMalloc((void**)&(spBlQmr->tau2),sizeof(double)*maxNumEvals*maxNumEvals);  
   spBlQmr->ldtau2 =maxNumEvals;                


   cudaMalloc((void**)&(spBlQmr->thita2),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldthita2=maxNumEvals;                   
   cudaMalloc((void**)&(spBlQmr->hta2),sizeof(double)*maxNumEvals*maxNumEvals);   
   spBlQmr->ldhta2  =maxNumEvals;                 
   cudaMalloc((void**)&(spBlQmr->zita2_),sizeof(double)*maxNumEvals*maxNumEvals); 
   spBlQmr->ldzita2_=maxNumEvals;                   
   cudaMalloc((void**)&(spBlQmr->zita2),sizeof(double)*maxNumEvals*maxNumEvals);  
   spBlQmr->ldzita2 =maxNumEvals;                  


   /* finding memory requirements */
   /* mem req for matvec*/
   size_t bufferSizeSpMM=0;
   cusparseCreateCsr(&(spBlQmr->descA),jd->matrix->dim,jd->matrix->dim,jd->matrix->nnz,
               jd->matrix->devCsrRows,jd->matrix->devCols,jd->matrix->devValuesD,
               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);

   cusparseCreateDnMat(&(spBlQmr->descw),dim,maxNumEvals,dim,spBlQmr->w,CUDA_R_64F,CUSPARSE_ORDER_COL);

   cusparseSpMM_bufferSize(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one,spBlQmr->descA,spBlQmr->descw,&zero,spBlQmr->descw,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,&bufferSizeSpMM);

   /* mem req for  qr(v2,0) */
   int Lwork = 0;
   int maxLwork = 0;
   // tau will be used also at qr(qq)
   cudaMalloc((void**)&spBlQmr->qrtau,sizeof(double)*2*maxNumEvals);
   cusolverDnDgeqrf_bufferSize(jd->gpuH->cusolverH,jd->matrix->dim,maxNumEvals,(double *)spBlQmr->v2,spBlQmr->ldv2,&Lwork);
   maxLwork = max(Lwork,maxLwork);
   cusolverDnDorgqr_bufferSize(jd->gpuH->cusolverH,jd->matrix->dim,maxNumEvals,maxNumEvals,
                        (double *)spBlQmr->v2,spBlQmr->ldv2,(double *)spBlQmr->qrtau,&Lwork);
   maxLwork = max(Lwork,maxLwork);

   /* mem req for  qr(qq) */
   Lwork = 0;
   // using the same tau
   //cudaMalloc((void**)&spBlQmr->tau,sizeof(double)*maxNumEvals);
   cusolverDnDgeqrf_bufferSize(jd->gpuH->cusolverH,2*maxNumEvals,2*maxNumEvals,(double *)spBlQmr->qq,spBlQmr->ldqq,&Lwork);
   maxLwork = max(Lwork,maxLwork);
   cusolverDnDorgqr_bufferSize(jd->gpuH->cusolverH,2*maxNumEvals,2*maxNumEvals,2*maxNumEvals,
                        (double *)spBlQmr->qq,spBlQmr->ldqq,spBlQmr->qrtau,&Lwork);
   maxLwork = max(Lwork,maxLwork);

   
   spBlQmr->lworkMemSpace = max(sizeof(double)*maxLwork,bufferSizeSpMM);
   cudaMalloc((void**)&(spBlQmr->workMemSpace),spBlQmr->lworkMemSpace);
   cudaMalloc((void**)&(spBlQmr->devInfo),sizeof(int));

//   exit(0);
   return;
}

void blQmrD_destroy(blQmrSpace *spBlQmr){

   cudaFree(spBlQmr->rin);
   cudaFree(spBlQmr->w);

   cudaFree(spBlQmr->v1);
   cudaFree(spBlQmr->v2);
   cudaFree(spBlQmr->v3);

   cudaFree(spBlQmr->p0);
   cudaFree(spBlQmr->p1);
   cudaFree(spBlQmr->p2);

   cudaFree(spBlQmr->qq);
   cudaFree(spBlQmr->q0);
   cudaFree(spBlQmr->q1);
   cudaFree(spBlQmr->q2);


   cudaFree(spBlQmr->alpha);
   cudaFree(spBlQmr->vita2);
   cudaFree(spBlQmr->vita3);
   cudaFree(spBlQmr->tau2_);
   cudaFree(spBlQmr->tau2);


   cudaFree(spBlQmr->thita2);
   cudaFree(spBlQmr->hta2);
   cudaFree(spBlQmr->zita2_);
   cudaFree(spBlQmr->zita2);


   cudaFree(spBlQmr->qrtau);
   cudaFree(spBlQmr->workMemSpace);


   return;
}


void blQmrD(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){

   // Solves the system:  (I-QQ')M(Q-QQ')X = B
   double one       =  1.0;
   double zero      =  0.0;
   double minus_one = -1.0;

   double normA = jd->normMatrix;
   tol   = tol*normA;
   
   blQmrSpace *sp = jd->spInnerSolver->spBlQmr;

   double *rin = (double*)sp->rin; int ldrin = sp->ldrin;   
   double *w   = (double*)sp->w;   int ldw   = sp->ldw;

   double *v1 = (double*)sp->v1; int ldv1 = sp->ldv1;
   double *v2 = (double*)sp->v2; int ldv2 = sp->ldv2;
   double *v3 = (double*)sp->v3; int ldv3 = sp->ldv3;

   double *p0 = (double*)sp->p0; int ldp0 = sp->ldp0;
   double *p1 = (double*)sp->p1; int ldp1 = sp->ldp1;
   double *p2 = (double*)sp->p2; int ldp2 = sp->ldp2;

   double *qq = (double*)sp->qq; int ldqq = sp->ldqq;
   double *q0 = (double*)sp->q0; int ldq0 = sp->ldq0;
   double *q1 = (double*)sp->q1; int ldq1 = sp->ldq1;
   double *q2 = (double*)sp->q2; int ldq2 = sp->ldq2;



   double *alpha = (double*)sp->alpha; int ldalpha = sp->ldalpha;
   double *vita2 = (double*)sp->vita2; int ldvita2 = sp->ldvita2;
   double *vita3 = (double*)sp->vita3; int ldvita3 = sp->ldvita3;
   double *tau2_ = (double*)sp->tau2_; int ldtau2_ = sp->ldtau2_;
   double *tau2  = (double*)sp->tau2 ; int ldtau2  = sp->ldtau2;

   double *thita2 = (double*)sp->thita2; int ldthita2 = sp->ldthita2; 
   double *hta2   = (double*)sp->hta2;   int ldhta2   = sp->ldhta2;
   double *zita2_ = (double*)sp->zita2_; int ldzita2_ = sp->ldzita2_; 
   double *zita2  = (double*)sp->zita2;  int ldzita2  = sp->ldzita2;


   cusparseSpMatDescr_t descA  = sp->descA;
   cusparseDnMatDescr_t descw  = sp->descw;


   double *qrtau  = (double*)sp->qrtau;

   size_t lworkMemSpace  = sp->lworkMemSpace;
   void   *workMemSpace  = sp->workMemSpace;
   int    *devInfo       = sp->devInfo;

   cudaMemset(X,0,sizeof(double)*ldX*numEvals);
   cudaMemset(v1,0,sizeof(double)*ldv1*numEvals);
   cudaMemset(p0,0,sizeof(double)*ldp0*numEvals);
   cudaMemset(p1,0,sizeof(double)*ldp1*numEvals);


   // q0 = I
   // q1 = I
   int BLOCK_DIM_X = 8;
   int BLOCK_DIM_Y = 8;
   dim3 blockDim(max(BLOCK_DIM_X,2*maxNumEvals), max(BLOCK_DIM_X,2*maxNumEvals));  
   dim3 gridDim((2*maxNumEvals + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (2*maxNumEvals + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   initIdentityGPU<<<gridDim, blockDim>>>(q0, ldq0,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q1, ldq1,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q2, ldq2,2*numEvals, 2*numEvals);

   // v2 = B;
   cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,B, ldB, &zero, v2, ldv2,v2,ldv2);


   //[v2,vita2] = qr(v2,0);
   cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, &one,v2, ldv2, &zero, v2, ldv2,w,ldw);

   cusolverDnDgeqrf(jd->gpuH->cusolverH,dim,numEvals,w,ldw,
               qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
   cusolverDnDorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,w,ldw,
               qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
   cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, &one,
               w, ldw,v2, ldv2, &zero,vita2, ldvita2);
   cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,
               w, ldw, &zero, v2, ldv2, v2, ldv2);

   //tau2_ = eye(s,s)*vita2;
   cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&one,
               vita2, ldvita2, &zero, tau2_, ldtau2_,tau2_,ldtau2_);

   //rin  = b;
   cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, B, ldB, &zero, rin, ldrin,rin,ldrin);

    
   double reig     = 1;
   double normrin  = 1;
   double normrinp = 1;
   
   int conv = 0;


   for (int k=2;k<maxIter;k++){
//   for (int k=2;k<3;k++){
      if(k>= maxIter){
          break;
      }

      /* w = v2-Q*(Q'*v2); */
      // alpha is unused here
      // alpha = Q'*v2
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim, &one, Q, ldQ, v2, ldv2, 
                  &zero,alpha, ldalpha);
       
      //w = Q*alpha;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, &one, Q, ldQ, alpha, ldalpha, 
                  &zero,w, ldw);
      // w = v2 - w ;
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, v2, ldv2, &minus_one, w, ldw,w,ldw);

      //w = A*w; 
      cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,descA,descw,&zero,descw,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);


      //w = w-Q*(Q'*w);
      // alpha = Q'*w
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim,
                &one, Q, ldQ, w, ldw, &zero,alpha, ldalpha);
       
      //v3 = Q*alpha; 
      // v3 is unused at this point
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, 
               &one, Q, ldQ, alpha, ldalpha, &zero,v3, ldv3);
      // w = w - v3 ;
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,
               &one, w, ldw, &minus_one, v3, ldv3, w,ldw);
      

      /* alpha = w'*v2; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, 
               dim, &one, w, ldw, v2, ldv2, &zero, alpha, ldalpha);


      /* w = w -v2*alpha - v1*vita2'; */
      // w = w - v2*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, &minus_one, v2, ldv2, alpha, ldalpha, 
                  &one, w, ldw);

      // w = w - v1*vita2';      
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals, &minus_one, v1, ldv1, vita2, ldvita2, 
                  &one, w, ldw);

      /* [v3,vita3] = qr(w,0); */
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,w, ldw, &zero, v3, ldv3, v3,ldv3);
      cusolverDnDgeqrf(jd->gpuH->cusolverH,dim,numEvals,
            v3,ldv3,qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
      cusolverDnDorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,
            v3,ldv3,qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, 
            numEvals, numEvals, dim, &one, v3, ldv3,w, ldw, &zero,vita3, ldvita3);
   



      /*if(norm(vita3) < tol)
          %return;
      end*/
      double normvita3;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&vita3[0+i*ldrin], 1, &normVec);
         if(i==0){
            normvita3 = normVec;
         }else{
            normvita3 = max(normvita3,normVec);
         }
      }

      /* thita2 = q0(s+1:end,1:s)'*vita2'; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals+0*ldq0], ldq0,vita2, ldvita2, &zero, thita2, ldthita2);


      /* hta2 = q1(1:s,1:s)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,1:s)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2'
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], ldq0,vita2, ldvita2, &zero, w, numEvals);
      // hta2 = q1(1:s,1:s)'*w
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
               &one, &q1[0+0*ldq1], ldq1, w, numEvals, &zero, hta2, ldhta2);
      // hta2 = hta2 + q1(s+1:end,1:s)'*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
               &one, &q1[numEvals+0*ldq1], ldq1, alpha, ldalpha, &one, hta2, ldhta2);


      /* zita2_ = q1(1:s,s+1:end)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,s+1:end)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2';
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], ldq0,vita2, ldvita2, &zero, w, numEvals);

      // zita2_ = q1(1:s,s+1:end)'*w
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[0+numEvals*ldq1], ldq1, w, numEvals, &zero, zita2_, ldzita2_);

      // zzita2_ = zita2_ + q1(s+1:end,s+1:end)'*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[numEvals+numEvals*ldq1], ldq1, alpha, ldalpha, &one, zita2_, ldzita2_);


      /*   qq = [zita2_ rand(s,s); vita3 rand(s,s)]; */
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&zero, 
                  qq, ldqq, &one, zita2_, ldzita2_, qq,ldqq);
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,
                  &zero, &qq[numEvals+0*ldqq], ldqq, &one, vita3, ldvita3, &qq[numEvals+0*ldqq],ldqq);
      curandGenerateNormalDouble(jd->gpuH->curandH, &qq[0+numEvals*ldqq], numEvals*2*maxNumEvals,0,1);

   
      /* [q2,zita2] = qr(qq); */
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, 2*numEvals, 2*numEvals,&one, qq, ldqq, &zero, q2, ldq2, q2,ldq2);
      cusolverDnDgeqrf(jd->gpuH->cusolverH,2*numEvals,2*numEvals,
                 q2,ldq2,qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
      cusolverDnDorgqr(jd->gpuH->cusolverH,2*numEvals,2*numEvals,2*numEvals,
                q2,ldq2,qrtau,(double*)workMemSpace,lworkMemSpace/sizeof(double),devInfo);
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, 2*numEvals, 2*numEvals, 2*numEvals,
                  &one, q2, ldq2, qq, ldqq,&zero,w, 2*numEvals);

      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals,
                &one,w, 2*numEvals, &zero, zita2, ldzita2, zita2,ldzita2);


      /* p2 =  v2-p1*hta2-p0*thita2; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p1, ldp1,hta2, ldhta2,&zero,w, ldw);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p0, ldp0, thita2, ldthita2, &one, w, ldw);
      
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, 
                  &one, v2, ldv2, &minus_one, w, ldw, p2,ldp2);


      /* p2 = p2/zita2; */
      cublasDtrsm(jd->gpuH->cublasH,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                              dim, numEvals,&one,zita2, maxNumEvals,p2, dim);

      /* tau2 = q2(1:s,1:s)'*tau2_; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one,q2, ldq2, tau2_, ldtau2_, &zero, tau2, ldtau2);

      /* x = x + p2*tau2; */
      cublasDgemm(jd->gpuH->cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, p2, ldp2, tau2, ldtau2, &one, X, ldX);

      /* rin = rin -(v3*vita3+v2*alpha+v1*vita2')*(v2'*(p2*tau2)); */
      // w = v3*vita3+v2*alpha+v1*vita2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,v3, ldv3, vita3, ldvita3, &zero, w, ldw);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, v2, ldv2, alpha, ldalpha, &one, w, ldw);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals,
                  &one, v1, ldv1, vita2, ldvita2, &one, w, ldw);
      
      // at this point alpha is not used anymore, we using it as a temporary mat
      // alpha = v2'*p2
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim,
                  &one, v2, ldv2, p2, ldp2, &zero, alpha, ldalpha);
      // alpha = alpha*tau2
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, alpha, ldalpha, tau2, ldtau2, &zero, alpha, ldalpha);

      // rin = rin - w*alpha;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &minus_one, w, ldw, alpha, ldalpha, &one, rin, ldrin);



      // rho = norm(rin);
      double rho;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, dim,&rin[0+i*ldrin], 1, &normVec);
         if(i==0){
            rho = normVec;
         }else{
            rho = max(rho,normVec);
         }
      }


      //printf("%e\n",rho);
      /* tau2_ = q2(1:s,s+1:end)'*tau2_; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, &q2[0+numEvals*ldq2], ldq2, tau2_, ldtau2_, &zero, tau2_, ldtau2_);
      

      /* if(max(vecnorm(tau2_)<= tol))
             return
         end
      */
      double normtau2_;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&tau2_[0+i*ldtau2_], 1, &normVec);
         if(i==0){
            normtau2_ = normVec;
         }else{
            normtau2_ = max(normtau2_,normVec);
         }
      }

      

      double normrinp2 = normrinp;
      normrinp = normrin;
      normrin = rho;
         
   
      /* brin = max(vecnorm(b'*rin)); */
      double brin;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &rin[0+i*ldrin], 1, &normVec);
         if(i==0){
            brin = normVec;
         }else{
            brin = max(brin,normVec);
         }
      }

   

      /* normx = max(vecnorm(x)); */
      double normx;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, dim,&X[0+i*ldX], 1, &normx);
         if(i==0){
            normx = normVec;
         }else{
            normx = max(normx,normVec);
         }
      }



      /* bx = max(vecnorm(b'*x)); */
      double bx;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &X[0+i*ldX], 1, &normVec);
         if(i==0){
            bx = normVec;
         }else{
            bx = max(bx,normVec);
         }
      }


      
      double g = sqrt(abs(normrin*normrin-brin*brin));
      double sin = sqrt(abs(normx*normx-bx*bx))/abs(bx);
      double vitain = abs(1-brin);
      

      double reigprev = reig;
      if(vitain < g*sin){
          reig = sqrt(abs(g*g+vitain*vitain))/(abs(bx)*sqrt(abs(1+normx*normx)));
      }else{
          reig = (g+vitain*normx)/(abs(bx)*(1+normx*normx));
      }
         

      if ((normrin < 10^(-1/2) && reig < tol) && conv==0){
             maxIter = k+5; 
             conv = 1;
      }
         

      if ((normrin < 10^(-1/2) && vitain*sin/(1+sin*sin) > tol/(2)) && conv==0){
             maxIter = k+5; 
             conv = 1;
      }         


      if (( normrin < vitain*sin/sqrt(1+sin*sin) ) && conv==0){
             maxIter = k+5; 
             conv =1;
      }         




      if (((normrin/(double)normrinp)*(normrin/(double)normrinp) > 1.0/(2-(normrinp/normrinp2)*(normrinp/normrinp2))) && conv==0){
             maxIter = k+5; 
             conv = 1;
      }         

      if((normrin/normrinp > pow((reig/reigprev),0.9)) && conv ==0){
            maxIter = k+5; 
            conv = 1;
      }         



      double *tmp;
      int     ldtmp;
      
      tmp = v1;  ldtmp = ldv1;
      v1  = v2;  ldv1  = ldv2;
      v2  = v3;  ldv2  = ldv3;
      v3  = tmp; ldv3  = ldtmp;


      tmp   = vita2; ldtmp = ldvita2;
      vita2 = vita3; ldvita2 = ldvita3;
      vita3 = tmp  ; ldvita3 = ldtmp;  
         
      tmp = p0;  ldtmp = ldp0;
      p0  = p1;  ldp0  = ldp1;
      p1  = p2;  ldp1  = ldp2;
      p2  = tmp; ldp2  = ldtmp;

      tmp = q0;  ldtmp = ldq0;
      q0  = q1;  ldq0  = ldq1;
      q1  = q2;  ldq1  = ldq2;
      q2  = tmp; ldq2  = ldtmp;
   

   } // end main for loop
   //exit(0);
}
//==============================
//==============================
//==============================
//==============================
//==============================
#if 0
   double *v1 = (double*)sp->v1;
   double *v0 = (double*)sp->v0;
   double *p0 = (double*)sp->p0;
   double *p1 = (double*)sp->p1;
   double *p2 = (double*)sp->p2;


   double *v     = (double*)sp->v;
   double *v2    = (double*)sp->v2;
   double *tau2_ = (double*)sp->tau2_;
   double *tau2  = (double*)sp->tau2;
   double *rin   = (double*)sp->rin;
   
   double *w     = (double*)sp->w;
   double *v3    = (double*)sp->v3;
   double *vita2 = (double*)sp->vita2;

   double *thita2 = (double*)sp->thita2;
   double *hta2   = (double*)sp->hta2;
   double *zita2_ = (double*)sp->zita2_;
   double *vita3  = (double*)sp->vita3;
//   double *vita3  =  &zita2_[0+2*numEvals];

   double *q2     = (double*)sp->q2;
   double *q1     = (double*)sp->q1;
   double *q0     = (double*)sp->q0;

   double *zita2 = (double*)sp->zita2;


   cudaMemset(v0,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(v1,0,sizeof(double)*dim*maxNumEvals); 

   cudaMemset(p0,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(p1,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(p2,0,sizeof(double)*dim*maxNumEvals);
/*
   cudaMemset(c0,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(c1,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(b1,0,sizeof(double)*maxNumEvals*maxNumEvals); 
   cudaMemset(b0,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(a0,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(a1,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(d0,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(d1,0,sizeof(double)*maxNumEvals*maxNumEvals);
*/

   cudaMemset(v,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(v2,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(tau2_,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(tau2,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(rin,0,sizeof(double)*dim*maxNumEvals); 
   
   cudaMemset(w,0,sizeof(double)*dim*maxNumEvals);
   cudaMemset(v3,0,sizeof(double)*dim*maxNumEvals); 
   cudaMemset(vita2,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(vita3,0,sizeof(double)*maxNumEvals*maxNumEvals);

   cudaMemset(thita2,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(hta2,0,sizeof(double)*maxNumEvals*maxNumEvals);
   cudaMemset(zita2_,0,sizeof(double)*maxNumEvals*maxNumEvals);
   // Using same mem space for orthoganlization [zita2_;vita3] 


   cudaMemset(q0,0,sizeof(double)*2*maxNumEvals*2*maxNumEvals);
   cudaMemset(q1,0,sizeof(double)*2*maxNumEvals*2*maxNumEvals);
   cudaMemset(q2,0,sizeof(double)*2*maxNumEvals*2*maxNumEvals);
   cudaMemset(zita2,0,sizeof(double)*maxNumEvals*maxNumEvals);


   // scalars for early stopping
   double reig     = 1.0;
   double normrin  = 1.0;
   double normrinp = 1.0;


   // a0 = I
   // a1 = I
   // d0 = I
   // d1 = I
   int BLOCK_DIM_X = 8;
   int BLOCK_DIM_Y = 8;
   dim3 blockDim(max(BLOCK_DIM_X,2*maxNumEvals), max(BLOCK_DIM_X,2*maxNumEvals));  
   dim3 gridDim((2*maxNumEvals + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (2*maxNumEvals + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   initIdentityGPU<<<gridDim, blockDim>>>(q0, 2*maxNumEvals,maxNumEvals, maxNumEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q1, 2*maxNumEvals,maxNumEvals, maxNumEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q2, 2*maxNumEvals,maxNumEvals, maxNumEvals);

   // qr workspace
   int qrLwork1,qrLwork2;
   double *qrTAU = (double*)sp->qrTAU ; 
   int qrLwork = sp->qrLwork;
   double *qrWorkspace = (double*)sp->qrWorkspace; 
   int *devInfo = (int*)sp->devInfo;

   double one       =  1.0;
   double zero      =  0.0;
   double minus_one = -1.0;


   // X = 0;
   cudaMemset(X,0,sizeof(double)*dim*numEvals);

   // v = b-A*X
   cudaMemcpy(v,B,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

   // [v2,vita2] = qr(v,0);
   cudaMemcpy(v2,v,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
   cusolverDnDgeqrf(jd->gpuH->cusolverH,dim,numEvals,v2,dim,qrTAU,qrWorkspace,qrLwork,devInfo);
   cudaDeviceSynchronize();
   cusolverDnDorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,v2,dim,qrTAU,qrWorkspace,qrLwork,devInfo);
   cudaDeviceSynchronize();
   cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, &one, v2, dim,v, dim, 
               &zero,vita2, maxNumEvals);

   // tau2_ = I*vita2 ; at this point a0 = I
   // a0 = q0(1:numEvals,1:numEvals)';
   cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
             &one, q0, 2*maxNumEvals, vita2, maxNumEvals,&zero,tau2_, maxNumEvals);

   //rin = B;
   cudaMemcpy(rin,B,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);

   double *Qv2; cudaMalloc((void**)&Qv2,sizeof(double)*maxNumEvals*maxNumEvals);
   double *QQv2; cudaMalloc((void**)&QQv2,sizeof(double)*dim*maxNumEvals);
   double *alpha; cudaMalloc((void**)&alpha,sizeof(double)*maxNumEvals*maxNumEvals);
   double *tmpMM; cudaMalloc((void**)&tmpMM,sizeof(double)*2*maxNumEvals*2*maxNumEvals);
   double *rintmp; cudaMalloc((void**)&rintmp,sizeof(double)*dim*maxNumEvals);
   double *rintmpsmall; cudaMalloc((void**)&rintmpsmall,sizeof(double)*maxNumEvals*maxNumEvals);

   int conv = 0;
   // this will be the main loop
   //for(int k=2;k<maxit;k++){}
   for(int k=2;k<15;k++){   
   //===============================   
   //===============================   
   //===============================   


      // w = v2-Q*Q'*v2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N,maxNumEvals, numEvals, dim,
                  &one,Q, ldQ,v2, dim,&zero,Qv2,maxNumEvals);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, maxNumEvals,
                  &one, Q, ldQ, Qv2, maxNumEvals, &zero,QQv2,dim);


      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,v2, dim,&minus_one, QQv2, dim,w,dim);


      /* w = A*w */
      cusparseSpMatDescr_t descA;
      cusparseDnMatDescr_t descW;
      size_t bufferSizeSpMM;
      void *bufferSpMM; 
    
      cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,descA,descW,&zero,descW,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,bufferSpMM);




      // w = w - QQ'w
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N,maxNumEvals, numEvals, dim,
                  &one,Q, ldQ, w, dim,&zero,Qv2,maxNumEvals);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, maxNumEvals,
                  &one, Q, ldQ, Qv2, maxNumEvals, &zero,QQv2,dim);


      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,w, dim,&minus_one, QQv2, dim,w,dim);




      // alpha = w'v2

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N,numEvals, numEvals, dim,
                  &one, w, dim, v2, dim, &zero,alpha,maxNumEvals);
      

      // w = w - v2*alpha - v1*vita2';
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &minus_one, v2, dim, alpha, maxNumEvals, &one, w,dim);

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T,dim, numEvals, numEvals,
                  &minus_one, v1, dim, vita2, maxNumEvals, &one, w,dim);




      // [v3,vita3] = qr(w,0);
      cudaMemcpy(v3,w,sizeof(double)*dim*numEvals,cudaMemcpyDeviceToDevice);
      cusolverDnDgeqrf(jd->gpuH->cusolverH,dim,numEvals,v3,dim,qrTAU,qrWorkspace,qrLwork,devInfo);
      cudaDeviceSynchronize();
      cusolverDnDorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,v3,dim,qrTAU,qrWorkspace,qrLwork,devInfo);
      cudaDeviceSynchronize();
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, &one, v3, dim,w, dim, 
                  &zero,vita3, maxNumEvals);





      // if(norm(vita3) < tol) return;
      double maxNorm=-1e+500;
      for(int i=0;i<numEvals;i++){
         double normV;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&vita3[0+i*maxNumEvals], 1, &normV);
         maxNorm = max(maxNorm,normV);
      }
      //if(maxNorm < tol){return;}

      // b0 = q(numEvals+1:end,1:numEvals)';
      //thita2 = q0(s+1:end,1:s)'*vita2';
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals + 0 * 2*maxNumEvals], 2*maxNumEvals, vita2, maxNumEvals,&zero,thita2, maxNumEvals);

      //hta2 = q1(1:s,1:s)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,1:s)'*alpha;
             
      // hta2 = q0(s+1:end,s+1:end)'*vita2'
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals+numEvals*2*maxNumEvals], 2*maxNumEvals, vita2, maxNumEvals,&zero,hta2, maxNumEvals);

      // hta2  = q1(1:s,1:s)'*hta2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, q1, 2*maxNumEvals, hta2, maxNumEvals, &zero, hta2, maxNumEvals);

      // hta2 = hta2 + q(s+1:end,1:s)'*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, &q1[numEvals + 0*2*maxNumEvals ], 2*maxNumEvals, alpha, maxNumEvals, &one, hta2, maxNumEvals);




      // zita2_ = q1(1:s,s+1:end)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,s+1:end)'*alpha;

      // zita2_ = q0(s+1:end,s+1:end)'*vita2';
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals+numEvals*2*maxNumEvals], 2*maxNumEvals, vita2, maxNumEvals,&zero,zita2_, maxNumEvals);

      //zita2_ = q1(1:s,s+1:end)'*zita2_
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, &q1[0+numEvals*2*maxNumEvals], 2*maxNumEvals, zita2_, maxNumEvals, &zero, zita2_ , maxNumEvals);


      // zita2_ = zita2_ + q1(s+1:end,s+1:end)'*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, &q1[numEvals + numEvals*2*maxNumEvals], 2*maxNumEvals, alpha, maxNumEvals, &one, zita2_, maxNumEvals);




      // [q,zita2] = qr([zita2_;vita3]); q = q';
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,
                             &one,zita2_,maxNumEvals,&zero,tmpMM, 2*maxNumEvals,tmpMM, 2*maxNumEvals);
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals, 
                     &one, vita3, maxNumEvals,&zero, &tmpMM[numEvals], 2*maxNumEvals,
                     &tmpMM[numEvals], 2*maxNumEvals);
      curandGenerateNormalDouble(jd->gpuH->curandH, tmpMM, numEvals*2*numEvals,0,1);

      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,2*numEvals, 2*numEvals,
                             &one,tmpMM,2*maxNumEvals,&zero,q2, 2*maxNumEvals,q2, 2*maxNumEvals);

      cusolverDnDgeqrf(jd->gpuH->cusolverH,2*numEvals,2*numEvals,q2,2*maxNumEvals,qrTAU,qrWorkspace,qrLwork,devInfo);
      cudaDeviceSynchronize();
      cusolverDnDorgqr(jd->gpuH->cusolverH,2*numEvals,2*numEvals,2*numEvals,q2,2*maxNumEvals,qrTAU,qrWorkspace,qrLwork,devInfo);
      cudaDeviceSynchronize();

      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, 2*numEvals, 2*numEvals, 2*numEvals,
                   &one, q2, 2*maxNumEvals,tmpMM, 2*maxNumEvals, &zero,tmpMM, 2*maxNumEvals);


      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,
                   &zero,zita2,maxNumEvals,&one,tmpMM, 2*maxNumEvals,zita2, maxNumEvals);

      /* at this point q = q' should be used */
      /* but instead we keep q' to use with tranposition at every operator */

      /* p2 =  v2-p1*hta2-p0*thita2; */
      // p2 = p1*hta2
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &one, p1, dim, hta2, maxNumEvals, &zero, p2, dim);

      // p2 = p2 + p0*thita2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &one, p0, dim, thita2, maxNumEvals, &one, p2, dim);
      
      // p2 = v2 - p2;   
      cublasDgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,
                   &one,v2,dim,&minus_one,p2, dim,p2, dim);

      /* p2 = p2/zita2; */
      cublasDtrsm(jd->gpuH->cublasH,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                              dim, numEvals,&one,zita2, maxNumEvals,p2, dim);
      /* tau2 = q(1:numEvals,1:numEvals)'*tau2_; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, q2, 2*maxNumEvals, tau2_, maxNumEvals, &zero, tau2, maxNumEvals);

      /* x = x + p2*tau2; */
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &one, p2, dim, tau2, maxNumEvals, &one, X, dim);
      
      /* rin22 = rin -(v3*vita3+v2*alpha+v1*vita2')*(v2'*p2)*tau2; */
     // rintmp =  v3*vita3
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &one, v3, dim, vita3, maxNumEvals, &zero, rintmp, dim);
      
      // rintmp = rintmp + v2*alpha
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &one, v2, dim, alpha, maxNumEvals, &one, rintmp, dim);

      // rintmp = rintmp + v1*vita2'
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals, 
                  &one, v1, dim, vita2, maxNumEvals, &one, rintmp, dim);


      // rintmpsmall = v2'*p2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, 
                  &one, v2, dim, p2, dim, &zero, rintmpsmall, maxNumEvals);

      // rintmpsmall = rintmpsmall*tau2;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, rintmpsmall, maxNumEvals, tau2, maxNumEvals, &zero, rintmpsmall, maxNumEvals);
      // rin = rin - rintmp*rintmpsmall
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &minus_one, rintmp, dim, rintmpsmall, maxNumEvals, &one, rin, dim);
      /* rho = norm(rin); */
      double rho;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&rin[0+i*dim], 1, &normVec);
         if(i==0){
            rho = normVec;
         }else{
            rho = max(rho,normVec);
         }
      }

      //if( rho < tol) return;   
printf("%% %e\n",rho);


      // ttau2_ = q(1:numEvals,numEvals+1:end)'*tau2_; PROSOXI: THELOYME TO TRANSPOSE STO q
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
                  &one, &q2[0 + numEvals*2*maxNumEvals], 2*maxNumEvals, tau2_, maxNumEvals, &zero, tau2_, maxNumEvals);



      
      double normtau2_;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&tau2_[0+i*maxNumEvals], 1, &normVec);
         if(i==0){
            normtau2_ = normVec;
         }else{
            normtau2_ = max(normtau2_,normVec);
         }
      }

      //if( normtau2_ < tol) return;   

      double normrinp2 = normrinp;
      normrinp = normrin;
      normrin = rho;
            

      /* brin = max(vecnorm(b'*rin)); */
      // rintmpsmall = B'*rin;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, 
                  &one, B, dim, rin, dim, &zero, rintmpsmall, maxNumEvals);

      double brin;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&rintmpsmall[0+i*maxNumEvals], 1, &normVec);
         if(i==0){
            brin = normVec;
         }else{
            brin = max(brin,normVec);
         }
      }
      


      /* normx = max(vecnorm(x)); */
      double normx;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&X[0+i*dim], 1, &normVec);
         if(i==0){
            normx = normVec;
         }else{
            normx = max(normx,normVec);
         }
      }

      /* bx = max(vecnorm(B'*X)); */
      // rintmpsmall = B'*X;
      cublasDgemm(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, 
                  &one, B, dim, X, dim, &zero, rintmpsmall, maxNumEvals);

      double bx;
      for(int i=0;i<numEvals;i++){
         double normVec;   
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&rintmpsmall[0+i*maxNumEvals], 1, &normVec);
         if(i==0){
            bx = normVec;
         }else{
            bx = max(bx,normVec);
         }
      }



      double g = sqrt(abs(normrin*normrin-brin*brin));
      double sin = sqrt(abs(normx*normx-bx*bx))/abs(bx);
      double vitain = abs(1-brin);
         
      double reigprev = reig;
      if(vitain < g*sin){
          reig = sqrt(abs(g*g+vitain*vitain))/(abs(bx)*sqrt(abs(1+normx*normx)));
      }else{
          reig = (g+vitain*normx)/(abs(bx)*(1+normx*normx));
      }

      if((normrin < 10^(-1/2) && reig < tol) && conv==0){
         //maxit = k+15 ;
         conv = 1;
         
      }
      
      if ((normrin < 10^(-1/2) && vitain*sin/(1+sin*sin) > tol/2) && conv==0){
         //maxit = k+15 ;
         conv = 1;
      }

            
      if (( normrin < vitain*sin/sqrt(1+sin*sin) ) && conv==0){
      //   maxit = k+15 ;
         conv =1;
      }


      if (((normrin/normrinp)*(normrin/normrinp) > 1/(2-(normrinp/normrinp2)*(normrinp/normrinp2))) && conv==0){
   //      maxit = k+15 ;
         conv = 1;
      }  

   //   double reig_reigprev = (double)reig/reigprev;
      if ((normrin/normrinp > pow(reig/reigprev,0.9)) && conv ==0){
   //      maxit = k+15 ;
         conv = 1;
      }


      double *tmp;
      tmp = v1;
      v1  = v2;
      v2  = v3;
      v3  = tmp;

      tmp   = vita2;
      vita2 = vita3;
      vita3 = tmp;

      tmp = p0;
      p0  = p1;
      p1  = p2;
      p2  = tmp;

      tmp = q0;
      q0  = q1;
      q1  = q2;
      q2  = tmp;
   }
   //===============================   
   //===============================   
   //===============================   

/*
   printMatrixDouble(c1,numEvals,numEvals,"c1");
   printMatrixDouble(d0,numEvals,numEvals,"d0");
   printMatrixDouble(vita2,numEvals,numEvals,"vita2");

   printMatrixDouble(d1,numEvals,numEvals,"d1");
   printMatrixDouble(alpha,numEvals,numEvals,"alpha");
*/
   exit(0);
   cudaFree(rintmpsmall);
   cudaFree(rintmp);
   cudaFree(tmpMM);
   cudaFree(alpha);
   cudaFree(QQv2);
   cudaFree(Qv2);
}
#endif


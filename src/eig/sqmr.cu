#include "../../include/jdqmr16.h"
#include "../include/helper.h"
#include <curand.h>
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "sqmr.h"
#include "../matrix/double2halfMat.h"
#include <math.h>
#include <omp.h>
__global__ void initIdentityGPU(double *devMatrix, int ldMat, int numR, int numC) {
    int r = blockDim.x*blockIdx.x + threadIdx.x;
    int c = blockDim.y*blockIdx.y + threadIdx.y;

    if(r < numR && c < numC) {
          if(r == c)
              devMatrix[r + c*ldMat] = 1;
          else
              devMatrix[r + c*ldMat] = 0;
    }
}

__global__ void initIdentityGPU(float *devMatrix, int ldMat, int numR, int numC) {
    int r = blockDim.x*blockIdx.x + threadIdx.x;
    int c = blockDim.y*blockIdx.y + threadIdx.y;

    if(r < numR && c < numC) {
          if(r == c)
              devMatrix[r + c*ldMat] = 1;
          else
              devMatrix[r + c*ldMat] = 0;
    }
}




void blQmrH_init(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){

   float one       =  1.0;
   float zero      =  0.0;
   float minus_one = -1.0;

   cublasSetMathMode(jd->gpuH->cublasH,CUBLAS_TF32_TENSOR_OP_MATH);

   cudaMalloc((void**)&(spBlQmr->w16),sizeof(half)*dim*maxNumEvals);   
   spBlQmr->ldw16  =dim;


   /* Create fp16 Matrix */
   /* find max-norm of matrix */
   jd->normMatrix = 0;
   double *val = jd->matrix->values;
   for(int i=0; i<jd->matrix->nnz; i++){
      if(abs(val[i]) > jd->normMatrix){
         jd->normMatrix = abs(val[i]);      
      }
   }

   /* Half precision matrix creation */
   double *vec; cudaMalloc((void**)&vec,(jd->matrix->nnz)*sizeof(double));
   cudaMemcpy(vec,jd->matrix->devValuesD,(jd->matrix->nnz)*sizeof(double),cudaMemcpyDeviceToDevice);
   double alpha; 
   if(jd->useHalf == USE_FP16){
      if(jd->normMatrix > 1e+03){
         alpha = (2048)/(jd->normMatrix);
         cublasScalEx(jd->gpuH->cublasH,jd->matrix->nnz,&alpha,CUDA_R_64F,vec,CUDA_R_64F,1,CUDA_R_64F);
      }
   }
   CUDA_CALL(double2halfMat(jd->matrix->devValuesH, jd->matrix->nnz, vec, jd->matrix->nnz, jd->matrix->nnz, 1));
   cudaFree(vec);

   /* finding memory requirements */
   /* mem req for matvec*/
   size_t bufferSizeSpMM=0;
   cusparseCreateCsr(&(spBlQmr->descA16),jd->matrix->dim,jd->matrix->dim,jd->matrix->nnz,
               jd->matrix->devCsrRows,jd->matrix->devCols,jd->matrix->devValuesH,
               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_16F);


   blQmrD_init(0, 0, 0, 0, 0, 0, dim, numEvals,maxNumEvals, tol, maxIter, spBlQmr,jd);

   return;
}

void blQmrH_destroy(blQmrSpace *spBlQmr){

   cudaFree(spBlQmr->w16);
   blQmrD_destroy(spBlQmr);
   return;
}

void blQmrH(float *X, int ldX,float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){

   // Solves the system:  (I-QQ')M(Q-QQ')X = B
   float one       =  1.0;
   float zero      =  0.0;
   float minus_one = -1.0;
   float thita     =  0.0;
   float normA = jd->normMatrix;
   tol   = tol*normA;
   
   blQmrSpace *sp = spBlQmr;
   cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
   cublasGemmAlgo_t    algo        = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

   float *rin  = (float*)sp->rin; int ldrin = sp->ldrin;   
   float *w    = (float*)sp->w;   int ldw   = sp->ldw;
   float *ww   = (float*)sp->ww;  int ldww  = sp->ldww;
   half  *w16  = (half *)sp->w16; int ldw16 = sp->ldw16;

   float *v1 = (float*)sp->v1; int ldv1 = sp->ldv1;
   float *v2 = (float*)sp->v2; int ldv2 = sp->ldv2;
   float *v3 = (float*)sp->v3; int ldv3 = sp->ldv3;

   float *p0 = (float*)sp->p0; int ldp0 = sp->ldp0;
   float *p1 = (float*)sp->p1; int ldp1 = sp->ldp1;
   float *p2 = (float*)sp->p2; int ldp2 = sp->ldp2;

   float *qq = (float*)sp->qq; int ldqq = sp->ldqq;
   float *q0 = (float*)sp->q0; int ldq0 = sp->ldq0;
   float *q1 = (float*)sp->q1; int ldq1 = sp->ldq1;
   float *q2 = (float*)sp->q2; int ldq2 = sp->ldq2;



   float *alpha = (float*)sp->alpha; int ldalpha = sp->ldalpha;
   float *vita2 = (float*)sp->vita2; int ldvita2 = sp->ldvita2;
   float *vita3 = (float*)sp->vita3; int ldvita3 = sp->ldvita3;
   float *tau2_ = (float*)sp->tau2_; int ldtau2_ = sp->ldtau2_;
   float *tau2  = (float*)sp->tau2 ; int ldtau2  = sp->ldtau2;

   float *thita2 = (float*)sp->thita2; int ldthita2 = sp->ldthita2; 
   float *hta2   = (float*)sp->hta2;   int ldhta2   = sp->ldhta2;
   float *zita2_ = (float*)sp->zita2_; int ldzita2_ = sp->ldzita2_; 
   float *zita2  = (float*)sp->zita2;  int ldzita2  = sp->ldzita2;


   cusparseCreateDnMat(&(spBlQmr->descw16),dim,numEvals,dim,spBlQmr->w16,CUDA_R_16F,CUSPARSE_ORDER_COL);

   size_t tmpbufferSizeSpMM;
   cusparseSpMM_bufferSize(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one,spBlQmr->descA16,spBlQmr->descw16,&zero,
                     spBlQmr->descw16,CUDA_R_16F,CUSPARSE_SPMM_CSR_ALG1,&tmpbufferSizeSpMM);

   cusparseSpMatDescr_t descA  = sp->descA16;
   cusparseDnMatDescr_t descw  = sp->descw16;


   float *qrtau  = (float*)sp->qrtau;

   size_t lworkMemSpace  = sp->lworkMemSpace;
   void   *workMemSpace  = sp->workMemSpace;
   int    *devInfo       = sp->devInfo;

   cudaMemset(X, 0,sizeof(float)*ldX *numEvals);
   cudaMemset(v1,0,sizeof(float)*ldv1*numEvals);
   cudaMemset(p0,0,sizeof(float)*ldp0*numEvals);
   cudaMemset(p1,0,sizeof(float)*ldp1*numEvals);


   // q0 = I
   // q1 = I
   int BLOCK_DIM_X = 8;//min(1024,4*numEvals);
   int BLOCK_DIM_Y = 8;
   int BLOCK_DIM_Z = 1 ; // min(3,4*numEvals/(BLOCK_DIM_X*BLOCK_DIM_Y);
   dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z);  
   dim3 gridDim(2*numEvals / blockDim.x + 1, 2*numEvals / blockDim.y + 1,1);
   initIdentityGPU<<<gridDim, blockDim>>>(q0, ldq0,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q1, ldq1,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q2, ldq2,2*numEvals, 2*numEvals);
   CUDA_CALL(cudaGetLastError());   


   // thita = Qmin'AQmin
   thita=0.0;
   float2halfMat(w16,ldw16,&Q[0+pivotThitaIdx*ldQ],ldQ,dim,1);
   cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,descA,descw,&zero,descw,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);
   half2floatMat(w,ldw,w16,ldw16,dim,numEvals);
   jd->numMatVecsfp16++;
   cublasSdot(jd->gpuH->cublasH, dim,&Q[0+pivotThitaIdx*ldQ],1,w, 1,&thita);
   // v2 = B;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,B, ldB, &zero, v2, ldv2,v2,ldv2);

   //[v2,vita2] = qr(v2,0);
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, &one,v2, ldv2, &zero, v2, ldv2,w,ldw);

   cusolverDnSgeqrf(jd->gpuH->cusolverH,dim,numEvals,w,ldw,
               qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
   cusolverDnSorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,w,ldw,
               qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
   cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, &one,
               w, CUDA_R_32F, ldw,v2,CUDA_R_32F, ldv2, &zero,vita2,CUDA_R_32F, ldvita2,
               computeType, algo);

   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,
               w, ldw, &zero, v2, ldv2, v2, ldv2);

   //tau2_ = eye(s,s)*vita2;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&one,
               vita2, ldvita2, &zero, tau2_, ldtau2_,tau2_,ldtau2_);

   //rin  = b;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, B, ldB, &zero, rin, ldrin,rin,ldrin);

   float reig     = 1;
   float normrin  = 1;
   float normrinp = 1;
   
   int conv = 0;
   for (int k=2;k<maxIter;k++){
      if(k>= maxIter){
          break;
      }

      /* w = v2-Q*(Q'*v2); */
      // alpha is unused here
      // alpha = Q'*v2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim,
                   &one, Q, CUDA_R_32F, ldQ, v2,CUDA_R_32F, ldv2, &zero,alpha,CUDA_R_32F, ldalpha,
                   computeType, algo);
      //w = Q*alpha;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, 
                  &one, Q, CUDA_R_32F, ldQ, alpha, CUDA_R_32F, ldalpha, &zero, w, CUDA_R_32F, ldw,
                   computeType, algo);

      // w = v2 - w ;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, v2, ldv2, &minus_one, w, ldw,w,ldw);

      // ww = thita*w;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,&thita, w, ldw, &zero, ww, ldww, ww,ldww);

      //w = A*w; // this step is done in fp16 
      float2halfMat(w16,ldw16,w,ldw,dim,numEvals);
      cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,descA,descw,&zero,descw,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);
      half2floatMat(w,ldw,w16,ldw16,dim,numEvals);
      jd->numMatVecsfp16++;

      // w = w-ww; (=w = w-thita w)
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,&one, w, ldw, &minus_one, ww, ldww, w,ldw);


      //w = w-Q*(Q'*w);
      // alpha = Q'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim,
                &one, Q, CUDA_R_32F, ldQ, w,CUDA_R_32F, ldw, &zero,alpha,CUDA_R_32F, ldalpha,
                computeType, algo);
       
      //v3 = Q*alpha; 
      // v3 is unused at this point
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, 
               &one, Q, CUDA_R_32F, ldQ, alpha, CUDA_R_32F,ldalpha, &zero,v3,CUDA_R_32F, ldv3,
                computeType, algo);
      // w = w - v3 ;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,
               &one, w, ldw, &minus_one, v3, ldv3, w,ldw);
      
      /* alpha = w'*v2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, 
               dim, &one, w, CUDA_R_32F, ldw, v2, CUDA_R_32F,ldv2, &zero, alpha,CUDA_R_32F, ldalpha,
                computeType, algo);


      /* w = w -v2*alpha - v1*vita2'; */
      // w = w - v2*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &minus_one, v2, CUDA_R_32F, ldv2, alpha,CUDA_R_32F, ldalpha, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);

      // w = w - v1*vita2';      
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals, 
                  &minus_one, v1, CUDA_R_32F, ldv1, vita2, CUDA_R_32F, ldvita2,&one, w,CUDA_R_32F, ldw,
                  computeType, algo);

      /* [v3,vita3] = qr(w,0); */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,w, ldw, &zero, v3, ldv3, v3,ldv3);
      cusolverDnSgeqrf(jd->gpuH->cusolverH,dim,numEvals,
            v3,ldv3,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      cusolverDnSorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,
            v3,ldv3,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim,
                     &one, v3, CUDA_R_32F, ldv3,w,CUDA_R_32F, ldw, &zero,vita3,CUDA_R_32F, ldvita3,
                     computeType, algo);
   


      /*if(norm(vita3) < tol)
          %return;
      end*/
      float normvita3=-1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, numEvals,&vita3[0+i*ldvita3], 1, &normVec);
         normvita3 = max(normvita3,normVec);
      }



      /* thita2 = q0(s+1:end,1:s)'*vita2'; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals+0*ldq0],CUDA_R_32F, ldq0,vita2,CUDA_R_32F, ldvita2, &zero, thita2,CUDA_R_32F, ldthita2,
                  computeType, algo);


      /* hta2 = q1(1:s,1:s)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,1:s)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2'
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], CUDA_R_32F, ldq0,vita2, CUDA_R_32F, ldvita2, &zero, w,CUDA_R_32F,  numEvals,
               computeType, algo);

      // hta2 = q1(1:s,1:s)'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
               &one, &q1[0+0*ldq1], CUDA_R_32F, ldq1, w,CUDA_R_32F, numEvals, &zero, hta2,CUDA_R_32F, ldhta2,
               computeType, algo);

      // hta2 = hta2 + q1(s+1:end,1:s)'*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
               &one, &q1[numEvals+0*ldq1], CUDA_R_32F, ldq1, alpha, CUDA_R_32F, ldalpha, &one, hta2, CUDA_R_32F, ldhta2,
               computeType, algo);


      /* zita2_ = q1(1:s,s+1:end)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,s+1:end)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2';
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], CUDA_R_32F, ldq0, vita2, CUDA_R_32F,ldvita2, &zero, w,CUDA_R_32F, numEvals,
               computeType, algo);

      // zita2_ = q1(1:s,s+1:end)'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[0+numEvals*ldq1], CUDA_R_32F, ldq1, w, CUDA_R_32F,numEvals, &zero, zita2_,CUDA_R_32F, ldzita2_,
                computeType, algo);

      // zzita2_ = zita2_ + q1(s+1:end,s+1:end)'*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[numEvals+numEvals*ldq1], CUDA_R_32F, ldq1, alpha, CUDA_R_32F,ldalpha, &one, zita2_,CUDA_R_32F, ldzita2_,
                computeType, algo);


      /*   qq = [zita2_ rand(s,s); vita3 rand(s,s)]; */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&zero, 
                  qq, ldqq, &one, zita2_, ldzita2_, qq,ldqq);
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,
                  &zero, &qq[numEvals+0*ldqq], ldqq, &one, vita3, ldvita3, &qq[numEvals+0*ldqq],ldqq);

      curandGenerateNormal(jd->gpuH->curandH, &qq[0+numEvals*ldqq], numEvals*2*maxNumEvals,0,1);

      /* [q2,zita2] = qr(qq); */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, 2*numEvals, 2*numEvals,&one, qq, ldqq, &zero, q2, ldq2, q2,ldq2);
      cusolverDnSgeqrf(jd->gpuH->cusolverH,2*numEvals,2*numEvals,
                 q2,ldq2,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      cusolverDnSorgqr(jd->gpuH->cusolverH,2*numEvals,2*numEvals,2*numEvals,
                q2,ldq2,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, 2*numEvals, 2*numEvals, 2*numEvals,
                  &one, q2, CUDA_R_32F, ldq2, qq,CUDA_R_32F, ldqq,&zero,w,CUDA_R_32F, 2*numEvals,
                  computeType, algo);

      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals,
                &one,w, 2*numEvals, &zero, zita2, ldzita2, zita2,ldzita2);


      /* p2 =  v2-p1*hta2-p0*thita2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p1, CUDA_R_32F, ldp1, hta2, CUDA_R_32F, ldhta2,&zero,w, CUDA_R_32F,ldw,
                  computeType, algo);

      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p0, CUDA_R_32F, ldp0, thita2,CUDA_R_32F, ldthita2, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);
      
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, 
                  &one, v2, ldv2, &minus_one, w, ldw, p2,ldp2);


      /* p2 = p2/zita2; */
      cublasStrsm(jd->gpuH->cublasH,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                              dim, numEvals,&one,zita2, maxNumEvals,p2, dim);

      /* tau2 = q2(1:s,1:s)'*tau2_; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one,q2, CUDA_R_32F, ldq2, tau2_,CUDA_R_32F, ldtau2_, &zero, tau2, CUDA_R_32F,ldtau2,
                  computeType, algo);

      /* x = x + p2*tau2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, p2, CUDA_R_32F, ldp2, tau2,CUDA_R_32F, ldtau2, &one, X,CUDA_R_32F, ldX,
                  computeType, algo);

      /* rin = rin -(v3*vita3+v2*alpha+v1*vita2')*(v2'*(p2*tau2)); */
      // w = v3*vita3+v2*alpha+v1*vita2;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,v3, CUDA_R_32F, ldv3, vita3,CUDA_R_32F, ldvita3, &zero, w,CUDA_R_32F, ldw,
                  computeType, algo);

      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, v2, CUDA_R_32F, ldv2, alpha,CUDA_R_32F, ldalpha, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);


      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals,
                  &one, v1, CUDA_R_32F, ldv1, vita2,CUDA_R_32F, ldvita2, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);
      
      // at this point alpha is not used anymore, we using it as a temporary mat
      // alpha = v2'*p2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim,
                  &one, v2, CUDA_R_32F, ldv2, p2,CUDA_R_32F, ldp2, &zero, alpha,CUDA_R_32F, ldalpha,
                  computeType, algo);

      // alpha = alpha*tau2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, alpha, CUDA_R_32F, ldalpha, tau2,CUDA_R_32F, ldtau2, &zero, alpha,CUDA_R_32F, ldalpha,
                  computeType, algo);

      // rin = rin - w*alpha;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &minus_one, w, CUDA_R_32F, ldw, alpha,CUDA_R_32F, ldalpha, &one, rin,CUDA_R_32F, ldrin,
                  computeType, algo);



      // rho = norm(rin);
      float rho = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, dim,&rin[0+i*ldrin], 1, &normVec);
         rho = max(rho,normVec);
      }
      //printf("%%%e\n",rho);

      /* tau2_ = q2(1:s,s+1:end)'*tau2_; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, &q2[0+numEvals*ldq2], CUDA_R_32F, ldq2, tau2_,CUDA_R_32F, ldtau2_, &zero, tau2_,CUDA_R_32F, ldtau2_,
                  computeType, algo);
      

      /* if(max(vecnorm(tau2_)<= tol))
             return
         end
      */
      float normtau2_ = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;
         cublasSnrm2(jd->gpuH->cublasH, numEvals,&tau2_[0+i*ldtau2_], 1, &normVec);
         normtau2_ = max(normtau2_,normVec);
         
      }

      

      float normrinp2 = normrinp;
      normrinp = normrin;
      normrin = rho;
         
   
      /* brin = max(vecnorm(b'*rin)); */
      float brin = -1e-10;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &rin[0+i*ldrin], 1, &normVec);
         brin = max(brin,normVec);
         
      }

   

      /* normx = max(vecnorm(x)); */
      float normx = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, dim,&X[0+i*ldX], 1, &normVec);
         normx = max(normx,normVec);
      }



      /* bx = max(vecnorm(b'*x)); */
      float bx;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &X[0+i*ldX], 1, &normVec);
         if(i==0){
            bx = normVec;
         }else{
            bx = max(bx,normVec);
         }
      }


      
      float g = sqrt(abs(normrin*normrin-brin*brin));
      float sin = sqrt(abs(normx*normx-bx*bx))/abs(bx);
      float vitain = abs(1-brin);
      

      float reigprev = reig;
      if(vitain < g*sin){
          reig = sqrt(abs(g*g+vitain*vitain))/(abs(bx)*sqrt(abs(1+normx*normx)));
      }else{
          reig = (g+vitain*normx)/(abs(bx)*(1+normx*normx));
      }
         

      if ((normrin < 10^(-1/2) && reig < tol) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }
         

      if ((normrin < 10^(-1/2) && vitain*sin/(1+sin*sin) > tol/(2)) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         


      if (( normrin < vitain*sin/sqrt(1+sin*sin) ) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         




      if (((normrin/(float)normrinp)*(normrin/(float)normrinp) > 1.0/(2-(normrinp/normrinp2)*(normrinp/normrinp2))) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         

      if((normrin/normrinp > pow((reig/reigprev),0.9)) && conv ==0){
            maxIter = ceil(1.25*k)+1;//+5; 
            conv = 1;
      }         



      float  *tmp;
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
}// end of function




















void blQmrF_init(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){


   float one       =  1.0;
   float zero      =  0.0;
   float minus_one = -1.0;

   cublasSetMathMode(jd->gpuH->cublasH,CUBLAS_TF32_TENSOR_OP_MATH);

   /* Create fp32 Matrix */
   /* find max-norm of matrix */
   jd->normMatrix = 0;
   double *val = jd->matrix->values;
   for(int i=0; i<jd->matrix->nnz; i++){
      if(abs(val[i]) > jd->normMatrix){
         jd->normMatrix = abs(val[i]);      
      }
   }

   /* Half precision matrix creation */
   double *vec; cudaMalloc((void**)&vec,(jd->matrix->nnz)*sizeof(double));
   cudaMemcpy(vec,jd->matrix->devValuesD,(jd->matrix->nnz)*sizeof(double),cudaMemcpyDeviceToDevice);
   double alpha; 
   if(jd->useHalf == USE_FP32){
      if(jd->normMatrix > 1e+04){
         alpha = (2048)/(jd->normMatrix);
         cublasScalEx(jd->gpuH->cublasH,jd->matrix->nnz,&alpha,CUDA_R_64F,vec,CUDA_R_64F,1,CUDA_R_64F);
      }
   }
   CUDA_CALL(double2floatMat(jd->matrix->devValuesF, jd->matrix->nnz, vec, jd->matrix->nnz, jd->matrix->nnz, 1));
   cudaDeviceSynchronize();
   cudaFree(vec);

   /* finding memory requirements */
   /* mem req for matvec*/
   size_t bufferSizeSpMM=0;
   cusparseCreateCsr(&(spBlQmr->descA16),jd->matrix->dim,jd->matrix->dim,jd->matrix->nnz,
               jd->matrix->devCsrRows,jd->matrix->devCols,jd->matrix->devValuesF,
               CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F);

   blQmrD_init(0, 0, 0, 0, 0, 0, dim, numEvals,maxNumEvals, 0.0, 0, spBlQmr,jd);
         
   return;
}

void blQmrF_destroy(blQmrSpace *spBlQmr){

   blQmrD_destroy(spBlQmr);
   return;
}


void blQmrF(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx,  float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){

   // Solves the system:  (I-QQ')M(Q-QQ')X = B
   float one       =  1.0;
   float zero      =  0.0;
   float minus_one = -1.0;

   float normA = jd->normMatrix;
   tol   = tol*normA;
   
   blQmrSpace *sp = jd->spInnerSolver->spBlQmr;
   cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
   cublasGemmAlgo_t    algo        = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

   float *rin  = (float*)sp->rin; int ldrin = sp->ldrin;   
   float *w    = (float*)sp->w;   int ldw   = sp->ldw;  
   float *ww   = (float*)sp->ww;  int ldww  = sp->ldww;

   float *v1 = (float*)sp->v1; int ldv1 = sp->ldv1;
   float *v2 = (float*)sp->v2; int ldv2 = sp->ldv2;
   float *v3 = (float*)sp->v3; int ldv3 = sp->ldv3;

   float *p0 = (float*)sp->p0; int ldp0 = sp->ldp0;
   float *p1 = (float*)sp->p1; int ldp1 = sp->ldp1;
   float *p2 = (float*)sp->p2; int ldp2 = sp->ldp2;

   float *qq = (float*)sp->qq; int ldqq = sp->ldqq;
   float *q0 = (float*)sp->q0; int ldq0 = sp->ldq0;
   float *q1 = (float*)sp->q1; int ldq1 = sp->ldq1;
   float *q2 = (float*)sp->q2; int ldq2 = sp->ldq2;



   float *alpha = (float*)sp->alpha; int ldalpha = sp->ldalpha;
   float *vita2 = (float*)sp->vita2; int ldvita2 = sp->ldvita2;
   float *vita3 = (float*)sp->vita3; int ldvita3 = sp->ldvita3;
   float *tau2_ = (float*)sp->tau2_; int ldtau2_ = sp->ldtau2_;
   float *tau2  = (float*)sp->tau2 ; int ldtau2  = sp->ldtau2;

   float *thita2 = (float*)sp->thita2; int ldthita2 = sp->ldthita2; 
   float *hta2   = (float*)sp->hta2;   int ldhta2   = sp->ldhta2;
   float *zita2_ = (float*)sp->zita2_; int ldzita2_ = sp->ldzita2_; 
   float *zita2  = (float*)sp->zita2;  int ldzita2  = sp->ldzita2;



   cusparseCreateDnMat(&(spBlQmr->descw16),dim,numEvals,dim,spBlQmr->w,CUDA_R_32F,CUSPARSE_ORDER_COL);
   size_t tmpbufferSizeSpMM;
   cusparseSpMM_bufferSize(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one,spBlQmr->descA16,spBlQmr->descw16,&zero,
                     spBlQmr->descw16,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG1,&tmpbufferSizeSpMM);


   cusparseSpMatDescr_t descA  = sp->descA16;
   cusparseDnMatDescr_t descw  = sp->descw16;
   

   float *qrtau  = (float*)sp->qrtau;

   size_t lworkMemSpace  = sp->lworkMemSpace;
   void   *workMemSpace  = sp->workMemSpace;
   int    *devInfo       = sp->devInfo;

   cudaMemset(X, 0,sizeof(float)*ldX *numEvals);
   cudaMemset(v1,0,sizeof(float)*ldv1*numEvals);
   cudaMemset(p0,0,sizeof(float)*ldp0*numEvals);
   cudaMemset(p1,0,sizeof(float)*ldp1*numEvals);


   // q0 = I
   // q1 = I
   int BLOCK_DIM_X = 8;//min(1024,4*numEvals);
   int BLOCK_DIM_Y = 8;
   int BLOCK_DIM_Z = 1 ; // min(3,4*numEvals/(BLOCK_DIM_X*BLOCK_DIM_Y);
   dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z);  
   dim3 gridDim(2*numEvals / blockDim.x + 1, 2*numEvals / blockDim.y + 1,1);
   initIdentityGPU<<<gridDim, blockDim>>>(q0, ldq0,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q1, ldq1,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q2, ldq2,2*numEvals, 2*numEvals);
   CUDA_CALL(cudaGetLastError());   

   // thita = Qmin'AQmin
   float thita=0.0;
   cudaMemcpy(w,&Q[0+pivotThitaIdx*ldQ],sizeof(float)*dim,cudaMemcpyDeviceToDevice);
   cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,descA,descw,&zero,descw,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);
   jd->numMatVecsfp16++;
   cublasSdot(jd->gpuH->cublasH, dim, &Q[0+pivotThitaIdx*ldQ],1,w, 1,&thita);
   //printf("thita: %e\n",thita);
   // v2 = B;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,B, ldB, &zero, v2, ldv2,v2,ldv2);

   //[v2,vita2] = qr(v2,0);
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, &one,v2, ldv2, &zero, v2, ldv2,w,ldw);

   cusolverDnSgeqrf(jd->gpuH->cusolverH,dim,numEvals,w,ldw,
               qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
   cusolverDnSorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,w,ldw,
               qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);

   //cublasSgemm
   cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim, &one,
               w, CUDA_R_32F, ldw,v2,CUDA_R_32F, ldv2, &zero,vita2,CUDA_R_32F, ldvita2,
               computeType, algo);

   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,
               w, ldw, &zero, v2, ldv2, v2, ldv2);

   //tau2_ = eye(s,s)*vita2;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&one,
               vita2, ldvita2, &zero, tau2_, ldtau2_,tau2_,ldtau2_);

   //rin  = b;
   cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, B, ldB, &zero, rin, ldrin,rin,ldrin);

    
   float reig     = 1;
   float normrin  = 1;
   float normrinp = 1;
   
   int conv = 0;
   for (int k=2;k<maxIter;k++){
      if(k>= maxIter){
          break;
      }

      /* w = v2-Q*(Q'*v2); */
      // alpha is unused here
      // alpha = Q'*v2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim,
                   &one, Q, CUDA_R_32F, ldQ, v2,CUDA_R_32F, ldv2, &zero,alpha,CUDA_R_32F, ldalpha,
                   computeType, algo);
       
      //w = Q*alpha;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, 
                  &one, Q, CUDA_R_32F, ldQ, alpha, CUDA_R_32F, ldalpha, &zero, w, CUDA_R_32F, ldw,
                   computeType, algo);

      // w = v2 - w ;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, &one, v2, ldv2, &minus_one, w, ldw,w,ldw);

      // ww = thita*w;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,&thita, w, ldw, &zero, ww, ldww, ww,ldww);

      //w = A*w; 
      cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,descA,descw,&zero,descw,CUDA_R_32F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);
      jd->numMatVecsfp16++;

      // w = w-ww; (=w = w-thita w)
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,&one, w, ldw, &minus_one, ww, ldww, w,ldw);

      //w = w-Q*(Q'*w);
      // alpha = Q'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, maxNumEvals, numEvals, dim,
                &one, Q, CUDA_R_32F, ldQ, w,CUDA_R_32F, ldw, &zero,alpha,CUDA_R_32F, ldalpha,
                computeType, algo);
       
      //v3 = Q*alpha; 
      // v3 is unused at this point
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, maxNumEvals, 
               &one, Q, CUDA_R_32F, ldQ, alpha, CUDA_R_32F,ldalpha, &zero,v3,CUDA_R_32F, ldv3,
                computeType, algo);
      // w = w - v3 ;
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals,
               &one, w, ldw, &minus_one, v3, ldv3, w,ldw);
      
      /* alpha = w'*v2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, 
               dim, &one, w, CUDA_R_32F, ldw, v2, CUDA_R_32F,ldv2, &zero, alpha,CUDA_R_32F, ldalpha,
                computeType, algo);


      /* w = w -v2*alpha - v1*vita2'; */
      // w = w - v2*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals, 
                  &minus_one, v2, CUDA_R_32F, ldv2, alpha,CUDA_R_32F, ldalpha, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);

      // w = w - v1*vita2';      
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals, 
                  &minus_one, v1, CUDA_R_32F, ldv1, vita2, CUDA_R_32F, ldvita2,&one, w,CUDA_R_32F, ldw,
                  computeType, algo);

      /* [v3,vita3] = qr(w,0); */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals,&one,w, ldw, &zero, v3, ldv3, v3,ldv3);
      cusolverDnSgeqrf(jd->gpuH->cusolverH,dim,numEvals,
            v3,ldv3,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      cusolverDnSorgqr(jd->gpuH->cusolverH,dim,numEvals,numEvals,
            v3,ldv3,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim,
                     &one, v3, CUDA_R_32F, ldv3,w,CUDA_R_32F, ldw, &zero,vita3,CUDA_R_32F, ldvita3,
                     computeType, algo);
   


      /*if(norm(vita3) < tol)
          %return;
      end*/
      float normvita3=-1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, numEvals,&vita3[0+i*ldvita3], 1, &normVec);
         normvita3 = max(normvita3,normVec);
      }



      /* thita2 = q0(s+1:end,1:s)'*vita2'; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
                  &one, &q0[numEvals+0*ldq0],CUDA_R_32F, ldq0,vita2,CUDA_R_32F, ldvita2, &zero, thita2,CUDA_R_32F, ldthita2,
                  computeType, algo);


      /* hta2 = q1(1:s,1:s)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,1:s)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2'
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], CUDA_R_32F, ldq0,vita2, CUDA_R_32F, ldvita2, &zero, w,CUDA_R_32F,  numEvals,
               computeType, algo);

      // hta2 = q1(1:s,1:s)'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
               &one, &q1[0+0*ldq1], CUDA_R_32F, ldq1, w,CUDA_R_32F, numEvals, &zero, hta2,CUDA_R_32F, ldhta2,
               computeType, algo);

      // hta2 = hta2 + q1(s+1:end,1:s)'*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals, 
               &one, &q1[numEvals+0*ldq1], CUDA_R_32F, ldq1, alpha, CUDA_R_32F, ldalpha, &one, hta2, CUDA_R_32F, ldhta2,
               computeType, algo);


      /* zita2_ = q1(1:s,s+1:end)'*q0(s+1:end,s+1:end)'*vita2' + q1(s+1:end,s+1:end)'*alpha; */
      // w = q0(s+1:end,s+1:end)'*vita2';
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_T, numEvals, numEvals, numEvals, 
               &one, &q0[numEvals+numEvals*ldq0], CUDA_R_32F, ldq0, vita2, CUDA_R_32F,ldvita2, &zero, w,CUDA_R_32F, numEvals,
               computeType, algo);

      // zita2_ = q1(1:s,s+1:end)'*w
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[0+numEvals*ldq1], CUDA_R_32F, ldq1, w, CUDA_R_32F,numEvals, &zero, zita2_,CUDA_R_32F, ldzita2_,
                computeType, algo);

      // zzita2_ = zita2_ + q1(s+1:end,s+1:end)'*alpha
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                &one, &q1[numEvals+numEvals*ldq1], CUDA_R_32F, ldq1, alpha, CUDA_R_32F,ldalpha, &one, zita2_,CUDA_R_32F, ldzita2_,
                computeType, algo);


      /*   qq = [zita2_ rand(s,s); vita3 rand(s,s)]; */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,&zero, 
                  qq, ldqq, &one, zita2_, ldzita2_, qq,ldqq);
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,numEvals, numEvals,
                  &zero, &qq[numEvals+0*ldqq], ldqq, &one, vita3, ldvita3, &qq[numEvals+0*ldqq],ldqq);

      curandGenerateNormal(jd->gpuH->curandH, &qq[0+numEvals*ldqq], numEvals*2*maxNumEvals,0,1);

      /* [q2,zita2] = qr(qq); */
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, 2*numEvals, 2*numEvals,&one, qq, ldqq, &zero, q2, ldq2, q2,ldq2);
      cusolverDnSgeqrf(jd->gpuH->cusolverH,2*numEvals,2*numEvals,
                 q2,ldq2,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      cusolverDnSorgqr(jd->gpuH->cusolverH,2*numEvals,2*numEvals,2*numEvals,
                q2,ldq2,qrtau,(float*)workMemSpace,lworkMemSpace/sizeof(float),devInfo);
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, 2*numEvals, 2*numEvals, 2*numEvals,
                  &one, q2, CUDA_R_32F, ldq2, qq,CUDA_R_32F, ldqq,&zero,w,CUDA_R_32F, 2*numEvals,
                  computeType, algo);

      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals,
                &one,w, 2*numEvals, &zero, zita2, ldzita2, zita2,ldzita2);


      /* p2 =  v2-p1*hta2-p0*thita2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p1, CUDA_R_32F, ldp1, hta2, CUDA_R_32F, ldhta2,&zero,w, CUDA_R_32F,ldw,
                  computeType, algo);

      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,p0, CUDA_R_32F, ldp0, thita2,CUDA_R_32F, ldthita2, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);
      
      cublasSgeam(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, 
                  &one, v2, ldv2, &minus_one, w, ldw, p2,ldp2);


      /* p2 = p2/zita2; */
      cublasStrsm(jd->gpuH->cublasH,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                              dim, numEvals,&one,zita2, maxNumEvals,p2, dim);

      /* tau2 = q2(1:s,1:s)'*tau2_; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one,q2, CUDA_R_32F, ldq2, tau2_,CUDA_R_32F, ldtau2_, &zero, tau2, CUDA_R_32F,ldtau2,
                  computeType, algo);

      /* x = x + p2*tau2; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, p2, CUDA_R_32F, ldp2, tau2,CUDA_R_32F, ldtau2, &one, X,CUDA_R_32F, ldX,
                  computeType, algo);

      /* rin = rin -(v3*vita3+v2*alpha+v1*vita2')*(v2'*(p2*tau2)); */
      // w = v3*vita3+v2*alpha+v1*vita2;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N,dim, numEvals, numEvals,
                  &one,v3, CUDA_R_32F, ldv3, vita3,CUDA_R_32F, ldvita3, &zero, w,CUDA_R_32F, ldw,
                  computeType, algo);

      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &one, v2, CUDA_R_32F, ldv2, alpha,CUDA_R_32F, ldalpha, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);


      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_T, dim, numEvals, numEvals,
                  &one, v1, CUDA_R_32F, ldv1, vita2,CUDA_R_32F, ldvita2, &one, w,CUDA_R_32F, ldw,
                  computeType, algo);
      
      // at this point alpha is not used anymore, we using it as a temporary mat
      // alpha = v2'*p2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, dim,
                  &one, v2, CUDA_R_32F, ldv2, p2,CUDA_R_32F, ldp2, &zero, alpha,CUDA_R_32F, ldalpha,
                  computeType, algo);

      // alpha = alpha*tau2
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, alpha, CUDA_R_32F, ldalpha, tau2,CUDA_R_32F, ldtau2, &zero, alpha,CUDA_R_32F, ldalpha,
                  computeType, algo);

      // rin = rin - w*alpha;
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_N, CUBLAS_OP_N, dim, numEvals, numEvals,
                  &minus_one, w, CUDA_R_32F, ldw, alpha,CUDA_R_32F, ldalpha, &one, rin,CUDA_R_32F, ldrin,
                  computeType, algo);



      // rho = norm(rin);
      float rho = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, dim,&rin[0+i*ldrin], 1, &normVec);
         rho = max(rho,normVec);
      }
      //printf("%%%e\n",rho);

      /* tau2_ = q2(1:s,s+1:end)'*tau2_; */
      //cublasSgemm
      cublasGemmEx(jd->gpuH->cublasH,CUBLAS_OP_T, CUBLAS_OP_N, numEvals, numEvals, numEvals,
                  &one, &q2[0+numEvals*ldq2], CUDA_R_32F, ldq2, tau2_,CUDA_R_32F, ldtau2_, &zero, tau2_,CUDA_R_32F, ldtau2_,
                  computeType, algo);
      

      /* if(max(vecnorm(tau2_)<= tol))
             return
         end
      */
      float normtau2_ = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;
         cublasSnrm2(jd->gpuH->cublasH, numEvals,&tau2_[0+i*ldtau2_], 1, &normVec);
         normtau2_ = max(normtau2_,normVec);
         
      }

      

      float normrinp2 = normrinp;
      normrinp = normrin;
      normrin = rho;
         
   
      /* brin = max(vecnorm(b'*rin)); */
      float brin = -1e-10;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &rin[0+i*ldrin], 1, &normVec);
         brin = max(brin,normVec);
         
      }

   

      /* normx = max(vecnorm(x)); */
      float normx = -1;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSnrm2(jd->gpuH->cublasH, dim,&X[0+i*ldX], 1, &normVec);
         normx = max(normx,normVec);
      }



      /* bx = max(vecnorm(b'*x)); */
      float bx;
      for(int i=0;i<numEvals;i++){
         float normVec;   
         cublasSdot(jd->gpuH->cublasH, dim, &B[0+i*ldB], 1, &X[0+i*ldX], 1, &normVec);
         if(i==0){
            bx = normVec;
         }else{
            bx = max(bx,normVec);
         }
      }


      
      float g = sqrt(abs(normrin*normrin-brin*brin));
      float sin = sqrt(abs(normx*normx-bx*bx))/abs(bx);
      float vitain = abs(1-brin);
      

      float reigprev = reig;
      if(vitain < g*sin){
          reig = sqrt(abs(g*g+vitain*vitain))/(abs(bx)*sqrt(abs(1+normx*normx)));
      }else{
          reig = (g+vitain*normx)/(abs(bx)*(1+normx*normx));
      }
         

      if ((normrin < 10^(-1/2) && reig < tol) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }
         

      if ((normrin < 10^(-1/2) && vitain*sin/(1+sin*sin) > tol/(2)) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         


      if (( normrin < vitain*sin/sqrt(1+sin*sin) ) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv =1;
      }         




      if (((normrin/(float)normrinp)*(normrin/(float)normrinp) > 1.0/(2-(normrinp/normrinp2)*(normrinp/normrinp2))) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         

      if((normrin/normrinp > pow((reig/reigprev),0.9)) && conv ==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         



      float  *tmp;
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
}// end of function






























void blQmrD_init(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd){


   double one       =  1.0;
   double zero      =  0.0;
   double minus_one = -1.0;

   size_t fpSize = sizeof(double);
   cudaMalloc((void**)&(spBlQmr->rin),fpSize*dim*maxNumEvals);
   spBlQmr->ldrin=dim;
   cudaMalloc((void**)&(spBlQmr->w),fpSize*dim*maxNumEvals);   
   spBlQmr->ldw  =dim;
   cudaMalloc((void**)&(spBlQmr->ww),fpSize*dim*maxNumEvals);   
   spBlQmr->ldww  =dim;

   cudaMalloc((void**)&(spBlQmr->v1),fpSize*dim*maxNumEvals); 
   spBlQmr->ldv1=dim;   
   cudaMalloc((void**)&(spBlQmr->v2),fpSize*dim*maxNumEvals); 
   spBlQmr->ldv2=dim;      
   cudaMalloc((void**)&(spBlQmr->v3),fpSize*dim*maxNumEvals); 
   spBlQmr->ldv3=dim;      

   cudaMalloc((void**)&(spBlQmr->p0),fpSize*dim*maxNumEvals); 
   spBlQmr->ldp0=dim;     
   cudaMalloc((void**)&(spBlQmr->p1),fpSize*dim*maxNumEvals); 
   spBlQmr->ldp1=dim;     
   cudaMalloc((void**)&(spBlQmr->p2),fpSize*dim*maxNumEvals); 
   spBlQmr->ldp2=dim;    

   cudaMalloc((void**)&(spBlQmr->qq),fpSize*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldqq=2*maxNumEvals;    
   cudaMalloc((void**)&(spBlQmr->q0),fpSize*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq0=2*maxNumEvals;       
   cudaMalloc((void**)&(spBlQmr->q1),fpSize*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq1=2*maxNumEvals;       
   cudaMalloc((void**)&(spBlQmr->q2),fpSize*2*maxNumEvals*2*maxNumEvals); 
   spBlQmr->ldq2=2*maxNumEvals;       


   cudaMalloc((void**)&(spBlQmr->alpha),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldalpha=maxNumEvals;          
   cudaMalloc((void**)&(spBlQmr->vita2),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldvita2=maxNumEvals;             
   cudaMalloc((void**)&(spBlQmr->vita3),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldvita3=maxNumEvals;                
   cudaMalloc((void**)&(spBlQmr->tau2_),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldtau2_=maxNumEvals;                
   cudaMalloc((void**)&(spBlQmr->tau2),fpSize*maxNumEvals*maxNumEvals);  
   spBlQmr->ldtau2 =maxNumEvals;                


   cudaMalloc((void**)&(spBlQmr->thita2),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldthita2=maxNumEvals;                   
   cudaMalloc((void**)&(spBlQmr->hta2),fpSize*maxNumEvals*maxNumEvals);   
   spBlQmr->ldhta2  =maxNumEvals;                 
   cudaMalloc((void**)&(spBlQmr->zita2_),fpSize*maxNumEvals*maxNumEvals); 
   spBlQmr->ldzita2_=maxNumEvals;                   
   cudaMalloc((void**)&(spBlQmr->zita2),fpSize*maxNumEvals*maxNumEvals);  
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
   cusolverDnDgeqrf_bufferSize(jd->gpuH->cusolverH,2*maxNumEvals,2*maxNumEvals,(double *)spBlQmr->qq,spBlQmr->ldqq,&Lwork);
   maxLwork = max(Lwork,maxLwork);
   cusolverDnDorgqr_bufferSize(jd->gpuH->cusolverH,2*maxNumEvals,2*maxNumEvals,2*maxNumEvals,
                        (double *)spBlQmr->qq,spBlQmr->ldqq,(double*)spBlQmr->qrtau,&Lwork);
   maxLwork = max(Lwork,maxLwork);

   
   spBlQmr->lworkMemSpace = max(sizeof(double)*maxLwork,bufferSizeSpMM);
   cudaMalloc((void**)&(spBlQmr->workMemSpace),spBlQmr->lworkMemSpace);
   cudaMalloc((void**)&(spBlQmr->devInfo),sizeof(int));

   return;
}

void blQmrD_destroy(blQmrSpace *spBlQmr){

   cudaFree(spBlQmr->rin);
   cudaFree(spBlQmr->w);
   cudaFree(spBlQmr->ww);

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
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx, double tol, int maxIter,
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
   double *ww  = (double*)sp->ww;  int ldww  = sp->ldww;

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


   cusparseCreateDnMat(&(spBlQmr->descw),dim,numEvals,dim,spBlQmr->w,CUDA_R_64F,CUSPARSE_ORDER_COL);
   size_t tmpbufferSizeSpMM;
   cusparseSpMM_bufferSize(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one,spBlQmr->descA,spBlQmr->descw,&zero,spBlQmr->descw,CUDA_R_64F,
                     CUSPARSE_SPMM_CSR_ALG1,&tmpbufferSizeSpMM);

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
   int BLOCK_DIM_X = 8;//min(1024,4*numEvals);
   int BLOCK_DIM_Y = 8;
   int BLOCK_DIM_Z = 1 ; // min(3,4*numEvals/(BLOCK_DIM_X*BLOCK_DIM_Y);
   dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z);  
   dim3 gridDim(2*numEvals / blockDim.x + 1, 2*numEvals / blockDim.y + 1,1);
   initIdentityGPU<<<gridDim, blockDim>>>(q0, ldq0,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q1, ldq1,2*numEvals, 2*numEvals);
   initIdentityGPU<<<gridDim, blockDim>>>(q2, ldq2,2*numEvals, 2*numEvals);
   CUDA_CALL(cudaGetLastError());   


   // thita = Qmin'AQmin
   double thita=0.0;
   cudaMemcpy(w,&Q[0+pivotThitaIdx*ldQ],sizeof(double)*dim,cudaMemcpyDeviceToDevice);
   cusparseSpMM(jd->gpuH->cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,descA,descw,&zero,descw,CUDA_R_64F,CUSPARSE_SPMM_CSR_ALG1,workMemSpace);
   jd->numMatVecsfp64++;
   cublasDdot(jd->gpuH->cublasH, dim, &Q[0+pivotThitaIdx*ldQ],1,w, 1,&thita);
//   printf("thita: %e %d\n",thita,pivotThitaIdx);


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
      jd->numMatVecsfp64++;

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
         cublasDnrm2(jd->gpuH->cublasH, numEvals,&vita3[0+i*ldvita3], 1, &normVec);
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
         cublasDnrm2(jd->gpuH->cublasH, dim,&X[0+i*ldX], 1, &normVec);
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
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }
         

      if ((normrin < 10^(-1/2) && vitain*sin/(1+sin*sin) > tol/(2)) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         


      if (( normrin < vitain*sin/sqrt(1+sin*sin) ) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv =1;
      }         




      if (((normrin/(double)normrinp)*(normrin/(double)normrinp) > 1.0/(2-(normrinp/normrinp2)*(normrinp/normrinp2))) && conv==0){
             maxIter = ceil(1.25*k)+1;//+5; 
             conv = 1;
      }         

      if((normrin/normrinp > pow((reig/reigprev),0.9)) && conv ==0){
             maxIter = ceil(1.25*k)+1;//+5; 
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
}// end of function


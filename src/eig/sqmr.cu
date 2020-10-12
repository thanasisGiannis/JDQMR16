#include "../../include/jdqmr16.h"
#include "../include/helper.h"


#include <curand.h>
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "sqmr.h"

void sqmr_init(half *X, int ldX, half *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd){

 
   half tmpScalar;
   half minus_one = __float2half(-1.0);
   half one  = __float2half(1.0);
   half zero = __float2half(0.0);



   half *x = X;
   half *b = B;
   
   struct gpuHandler               *gpuH   = jd->gpuH;
   cublasHandle_t                cublasH   = gpuH->cublasH;
   cusparseHandle_t              cusparseH = gpuH->cusparseH;
   struct innerSolverSpace       *spInnerSolver = jd->spInnerSolver;         
   struct sqmrSpace              *spsqmr   = spInnerSolver->spSQmr;


   half *t = x;
   cudaMalloc((void**)&(spsqmr->delta),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->r),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->d),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->w),sizeof(half)*dim);

   /* cusparse data initilization */
   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseCreateCoo(&(spsqmr->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesH,
							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_16F);


	cusparseCreateDnVec(&(spsqmr->descrd),dim,(void*)spsqmr->d,CUDA_R_16F);
	cusparseCreateDnVec(&(spsqmr->descrw),dim,(void*)spsqmr->w,CUDA_R_16F);

   cusparseSpMV_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spsqmr->descrA,spsqmr->descrd,&zero,
                        spsqmr->descrw,CUDA_R_16F,CUSPARSE_COOMV_ALG,&spsqmr->bufferSize);



   assert(spsqmr->bufferSize>=0);
   cudaMalloc((void**)&spsqmr->buffer,spsqmr->bufferSize);


}

void sqmr_destroy(struct jdqmr16Info *jd){

   struct innerSolverSpace       *spInnerSolver = jd->spInnerSolver;         
   struct sqmrSpace              *spsqmr   = spInnerSolver->spSQmr;


   cudaFree(spsqmr->buffer);
   cudaFree(spsqmr->w);
   cudaFree(spsqmr->delta);
   cudaFree(spsqmr->r);
   cudaFree(spsqmr->d);

}

void sqmr(half *X, int ldX, half *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd){

/*
   Solving AX=B with sQMR and early stopping criteria

*/

   // this should be input in this function
   float ita    = 0.0;
   float thita_ = 0.0; 
   int qmrMaxIt = 1000;
   int tol      = 1e-08;

   
   float Thita_ = 0.0;
   float rho_;
   float sigma;
   float alpha;  
   float normr;
   float Thita;
   float c;
   float g_;
   float r00;
   float rho;
   float vita;
   float g;
   float gama;
   float xi;
   float normt; 
   float f;
   float p;
   float thita;
   float pk;
   float rkm;
   float scalw;
      

   float minus_alpha;
   float deltaScal1;
   float deltaScal2;
   float one32 = 1.0;
      
   float BITA  = 0.0;
   float DELTA = 0.0;
   float GAMA  = 0.0;
   float FI    = 0.0;
   float PSI   = 0.0;


   half tmpScalar;
   half minus_one = __float2half(-1.0);
   half one  = __float2half(1.0);
   half zero = __float2half(0.0);



   half *x = X;
   half *b = B;
   
   struct gpuHandler               *gpuH   = jd->gpuH;
   cublasHandle_t                cublasH   = gpuH->cublasH;
   cusparseHandle_t              cusparseH = gpuH->cusparseH;
   struct innerSolverSpace       *spInnerSolver = jd->spInnerSolver;         
   struct sqmrSpace              *spsqmr   = spInnerSolver->spSQmr;

   cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST);
   cudaDeviceSynchronize();

#if 0
   half *t = x;
   cudaMalloc((void**)&(spsqmr->delta),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->r),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->d),sizeof(half)*dim);
   cudaMalloc((void**)&(spsqmr->w),sizeof(half)*dim);

   half *delta = (half*)spsqmr->delta;
   half *r = (half*)spsqmr->r;     
   half *d = (half*)spsqmr->d;     
   half *w = (half*)spsqmr->w;     

   /* cusparse data initilization */
   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseCreateCoo(&(spsqmr->descrA),dim,dim,A->nnz,A->devRows,A->devCols,A->devValuesH,
							CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_16F);


	cusparseCreateDnVec(&(spsqmr->descrd),dim,(void*)d,CUDA_R_16F);
	cusparseCreateDnVec(&(spsqmr->descrw),dim,(void*)w,CUDA_R_16F);

   cusparseSpMV_bufferSize(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,spsqmr->descrA,spsqmr->descrd,&zero,
                        spsqmr->descrw,CUDA_R_16F,CUSPARSE_COOMV_ALG,&spsqmr->bufferSize);


   cusparseSpMatDescr_t descrA = spsqmr->descrA;
   cusparseDnVecDescr_t descrd = spsqmr->descrd;
   cusparseDnVecDescr_t descrw = spsqmr->descrw;

   size_t bufferSize = spsqmr->bufferSize;// = spsqmr->bufferSize;
   void *buffer = spsqmr->buffer;// = spsqmr->buffer;

   assert(spsqmr->bufferSize>=0);
   cudaMalloc((void**)&spsqmr->buffer,spsqmr->bufferSize);

#else


   half *t = x;
   half *delta = (half*)spsqmr->delta;
   half *r = (half*)spsqmr->r;     
   half *d = (half*)spsqmr->d;     
   half *w = (half*)spsqmr->w;     

   /* cusparse data initilization */
   struct jdqmr16Matrix  *A = jd->matrix;
   cusparseSpMatDescr_t descrA = spsqmr->descrA;
   cusparseDnVecDescr_t descrd = spsqmr->descrd;
   cusparseDnVecDescr_t descrw = spsqmr->descrw;

   size_t bufferSize = spsqmr->bufferSize;// = spsqmr->bufferSize;
   void *buffer = spsqmr->buffer;// = spsqmr->buffer;


#endif

   /* r = -b */
   cudaMemcpy(r,b,sizeof(half)*dim,cudaMemcpyDeviceToDevice);
   cublasScalEx(cublasH,dim,&minus_one,CUDA_R_16F,r,CUDA_R_16F,1,CUDA_R_32F);

   /* d = r */
   cudaMemcpy(d,r,sizeof(half)*dim,cudaMemcpyDeviceToDevice);
   cublasNrm2Ex(cublasH,dim,b,CUDA_R_16F,1,&tmpScalar,CUDA_R_16F,CUDA_R_32F);
   g = __half2float(tmpScalar);


   cublasDotEx(cublasH,dim,(void*)r,CUDA_R_16F,1,(void*)d,CUDA_R_16F,1,(void*)&tmpScalar,CUDA_R_16F,CUDA_R_32F);
   rho_ = __half2float(tmpScalar);

   /* sQMR main iteration */
   for(int i=0; i<qmrMaxIt; i++){
      
      /* w = A*d */
      cusparseSpMV(cusparseH,CUSPARSE_OPERATION_NON_TRANSPOSE,
             &one,descrA,descrd,&zero,descrw,CUDA_R_16F,
             CUSPARSE_COOMV_ALG,buffer);


      /* sigma = d'*w */
      CUBLAS_CALL(cublasDotEx(cublasH,dim,d,CUDA_R_16F,1,w,CUDA_R_16F,1,&tmpScalar,CUDA_R_16F,CUDA_R_32F));
      sigma = __half2float(tmpScalar);
      /* alpha = rho_/sigma */
      alpha =1*rho_/sigma;
   

      /* r = r -alpha*w */
      minus_alpha = -1*alpha;
      CUBLAS_CALL(cublasAxpyEx(cublasH,dim,&minus_alpha,CUDA_R_32F,w,CUDA_R_16F,1,r,CUDA_R_16F,1,CUDA_R_32F));

      /* Thita = norm(r)/g */
      CUBLAS_CALL(cublasNrm2Ex(cublasH,dim,r,CUDA_R_16F,1,&tmpScalar,CUDA_R_16F,CUDA_R_32F));
      normr = __half2float(tmpScalar);
      Thita = normr/g;

      /* c = 1./sqrt(1+Thita*Thita) */
      c = sqrt(1/(1+Thita*Thita));
      /* g = g*Thita*c */
      g =g*Thita*c;

      if(i == 0){
         g_ = g;
      }

      /* delta = (c*c*alpha)*d + (c*c*Thita_*Thita_)*delta  */
      deltaScal1 = c*c*Thita_*Thita_;   
      deltaScal2 = c*c*alpha;
      CUBLAS_CALL(cublasScalEx(cublasH,dim,&deltaScal1,CUDA_R_32F,delta,CUDA_R_16F,1,CUDA_R_32F));
      CUBLAS_CALL(cublasAxpyEx(cublasH,dim,&deltaScal2,CUDA_R_32F,d,CUDA_R_16F,1,delta,CUDA_R_16F,1,CUDA_R_32F));
      /*  t  = t + delta */
      
      CUBLAS_CALL(cublasAxpyEx(cublasH,dim,&one32,CUDA_R_32F,delta,CUDA_R_16F,1,t,CUDA_R_16F,1,CUDA_R_32F));

      if(abs(g) < tol){
         break;
      }

      gama = c*c*Thita_; 
      xi = c*c*alpha;    
      normt; 
      cublasNrm2Ex(cublasH,dim,r,CUDA_R_16F,1,&tmpScalar,CUDA_R_16F,CUDA_R_32F); 
      normt = __half2float(tmpScalar);
      f = 1 + normt*normt; 
      PSI = gama*(PSI + FI);
      FI = gama*gama*FI + xi*xi*sigma;
      GAMA = GAMA  + 2*PSI + FI;


      DELTA = gama*DELTA - xi*rho_;
      BITA = BITA + DELTA;
      p = ((thita_-ita+2*BITA+GAMA))/f;
      thita = ita+p;

      pk = (((thita_)-ita+BITA)*((thita_)-ita+BITA))/f - p*p;
      rkm = sqrt(((g)*(g))/f + pk);


      if(i==0){
         r00 = rkm;
      }


      if(rho_ < tol){
         break;
      }


      rkm = sqrt(g*g/f);

       if( (g < rkm*max(0.99 * sqrt(f),sqrt(g/g_))) || (thita > thita_) || rkm<0.1*r00  || g < tol || rkm < tol){
          break; 
       }

      /*  w = r./norm(r); */
      cudaMemcpy(w,r,sizeof(half)*dim,cudaMemcpyDeviceToDevice);
      cublasNrm2Ex(cublasH,dim,w,CUDA_R_16F,1,&tmpScalar,CUDA_R_16F,CUDA_R_32F); 
      scalw = 1/__half2float(tmpScalar);
      cublasScalEx(cublasH,dim,&scalw,CUDA_R_32F,w,CUDA_R_16F,1,CUDA_R_32F);

         
      /* rho = r'*w; */
      cublasDotEx(cublasH,dim,(void*)r,CUDA_R_16F,1,(void*)w,CUDA_R_16F,1,(void*)&tmpScalar,CUDA_R_16F,CUDA_R_32F);
      rho = __half2float(tmpScalar);
      vita = rho/rho_;



      /* d = w + vita*d; */
      cublasScalEx(cublasH,dim,&vita,CUDA_R_32F,d,CUDA_R_16F,1,CUDA_R_32F);
      CUBLAS_CALL(cublasAxpyEx(cublasH,dim,&one32,CUDA_R_32F,w,CUDA_R_16F,1,d,CUDA_R_16F,1,CUDA_R_32F));


      thita_ = thita;
      Thita_ = Thita;
      rho_ = rho;
      g_ = g;

   }

#if 0
   cudaFree(spsqmr->buffer);
   cudaFree(spsqmr->w);
   cudaFree(spsqmr->delta);
   cudaFree(spsqmr->r);
   cudaFree(spsqmr->d);
#endif
}

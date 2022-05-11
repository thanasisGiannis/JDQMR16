#ifndef JDQMR16_H
#define JDQMR16_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <unistd.h> 
#include <curand.h>
#include <cusolverDn.h>

struct jdqmr16Matrix {
	/* cpu matrix space */
   double  *values;
	int     *cols;
	int     *rows;

	/* gpu matrix space */
   double  *devValuesD;
   double  *devColumnNorms;

 	half    *devValuesH;
	int     *devCols;
	int     *devRows;

	/* matrix general info */
	int dim;
	int nnz; 
};

struct gpuHandler{

   curandGenerator_t   curandH;
   cusolverDnHandle_t  cusolverH;
   cublasHandle_t      cublasH;
   cusparseHandle_t    cusparseH;
};

struct devSolverSpace{
   
   double *Vprev; int ldVprev; // previous Ritz vectors
   double *V;     int ldV;     // ritz vectors
   double *L;                  // ritz values
   double *W;     int ldW;     // space of projection (in the JD iteration)
   double *H;     int ldH;     // projected matrix
   double *R;     int ldR;     // EigenPair Residual Vectors

   double *AW;    int ldAW;    // keeping AW for fast update of basis
   double *P;     int ldP;     // P to expand basis


   int *lockedVals;  // binary matrix which points which evals are converged 
};


struct sqmrSpace{

   
   void *delta; 
   void *r;     
   void *d;     
   void *w;     
   size_t bufferSize;
   void *buffer;
   
   void *VTd;
   
   
   cusparseSpMatDescr_t descrA;
   cusparseDnVecDescr_t descrd;
   cusparseDnVecDescr_t descrw;


};

struct innerSolverSpace{

   double *B = NULL;    int ldB;
   double *VTB = NULL;  int ldVTB;
   double *X = NULL;    int ldX;      
   double *maxB = NULL; 
   int    *normIndexB = NULL;

   void *X16 = NULL;
   void *B16 = NULL;
   void *V16 = NULL;  int ldV16;
   
   struct sqmrSpace        *spSQmr;

};

struct restartSpace{

   double *d_tau;  
   int    *devInfo;
   double *d_work;
   int lwork = 0;

   cusparseSpMatDescr_t descrA;
   cusparseDnMatDescr_t descrW;
   cusparseDnMatDescr_t descrAW;

   double *AW; 
   int     ldAW;

   size_t  bufferSize;
   double *buffer;
     
};

struct expandBasisSpace{
   
   double *AP; int ldAP;

   /* P = P - W*W'*P */
   double *WTP = NULL; int ldWTP;

   /* P = orth(P) */
   double *d_tau;
   int    *devInfo;
   double *d_work;
   int lwork;

   /* AP = A*P */
   cusparseSpMatDescr_t descrA;
   cusparseDnMatDescr_t descrP;
   cusparseDnMatDescr_t descrAP;
   void *buffer; size_t bufferSize;

};


struct residualSpace{

   //double *AV = NULL; int ldAV;
   double *VL = NULL; int ldVL;
   double *hL = NULL;
   cusparseSpMatDescr_t descrA;
   cusparseDnMatDescr_t descrV;
   cusparseDnMatDescr_t descrR;

   void *buffer; size_t bufferSize;

};

struct eigHSpace{
   int     lwork;   
   int    *devInfo = NULL;
   double *d_work;
   double *QH;    int ldQH;   
   double *LH;
};



struct initBasisSpace{
   double *d_tau   = NULL;
   int    *devInfo = NULL;
   double *d_work  = NULL;
   double *d_R     = NULL;

   int lwork_geqrf ;
   int lwork_orgqr ;
   int lwork       ;

   cusparseSpMatDescr_t descrA = NULL;
	cusparseDnMatDescr_t descrV = NULL;
	cusparseDnMatDescr_t descrAV = NULL;
   size_t   bufferSize = 0;
   size_t   bufferSizeTrans = 0;
   void    *externalBuffer;

   double  *AV; int ldAV;
};

struct jdqmr16Info {
//   double *devL;
//   double *devQ;

//   double *L;
//   double *Q;

   int numEvals =  1;
   int maxBasis =  15;
   int maxIter  =  1000;
   double tol      =  1e-04;
   double normMatrix = 1;
   struct jdqmr16Matrix  *matrix;
   struct devSolverSpace *sp;
   struct gpuHandler     *gpuH;

   struct initBasisSpace   *spInitBasis;
   struct eigHSpace        *spEigH;
   struct residualSpace    *spResidual;
   struct expandBasisSpace *spExpandBasis;
   struct restartSpace     *spRestart;
   struct innerSolverSpace *spInnerSolver;

   int    numMatVecsfp64  = 0;
   int    numMatVecsfp16  = 0;
   int    outerIterations = 0;
   int    innerIterations = 0;

   int    useHalf = 1;
};

void init_jdqmr16(struct jdqmr16Info *jd);
void destroy_jdqmr16(struct jdqmr16Info *jd);
void jdqmr16(struct jdqmr16Info *jd);
void jdqmr16_eigenpairs(double *V, int ldV, double *L, struct jdqmr16Info *jd);

#endif


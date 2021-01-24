#ifndef JDQMR16_H
#define JDQMR16_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <unistd.h> 
#include <curand.h>
#include <cusolverDn.h>

#define USE_FP16  1
#define USE_FP32 -1
#define USE_FP64  0



struct lockSpace{
   
   double *QTV; int ldQTV;
   double *QTR; int ldQTR; 
   double *PR;  int ldPR;

   double *Lh;
   double *Llockedh;

};


struct devSolverSpace{
   
   double *Vprev;   int ldVprev;   // previous Ritz vectors
   double *V;       int ldV;       // ritz vectors
   double *L;                      // ritz values
   double *W;       int ldW;       // space of projection (in the JD iteration)
   double *H;       int ldH;       // projected matrix
   double *R;       int ldR;       // EigenPair Residual Vectors
   double *QH;      int ldQH;      // EigenVectors of the projected system

   double *AW;      int ldAW;      // keeping AW for fast update of basis
   double *P;       int ldP;       // P to expand basis

   double *normr;                  // residual of the eigenpairs
   int maxLockedVals;
   double *Llocked;                // Locked eigenvalues
   int     numLocked;              // number of locked eigenpairs 

};


struct blQmrSpace{

   void *rin; int ldrin;   
   void *w;   int ldw;

   void *v1; int ldv1;
   void *v2; int ldv2;
   void *v3; int ldv3;

   void *p0; int ldp0;
   void *p1; int ldp1;
   void *p2; int ldp2;

   void *qq; int ldqq;
   void *q0; int ldq0;
   void *q1; int ldq1;
   void *q2; int ldq2;



   void *alpha; int ldalpha;
   void *vita2; int ldvita2;
   void *vita3; int ldvita3;
   void *tau2_; int ldtau2_;
   void *tau2;  int ldtau2;

   void *thita2; int ldthita2; 
   void *hta2;   int ldhta2;
   void *zita2_; int ldzita2_; 
   void *zita2;  int ldzita2;


   cusparseSpMatDescr_t descA;
   cusparseDnMatDescr_t descw;


   double *qrtau;

   size_t lworkMemSpace;
   void   *workMemSpace;
   int    *devInfo;

};

struct innerSolverSpace{

   double *B;    int ldB;
   double *VTB;  int ldVTB;
   double *X;    int ldX;      
   double *maxB; 
   int    *normIndexB;

   void *P16;
   void *B16;
   void *V16;


   void *P32;
   void *B32;
   void *V32;
  
   struct blQmrSpace       *spBlQmr;
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
   size_t   bufferSize = -1;
   size_t   bufferSizeTrans = -1;
   void    *externalBuffer;

   double  *AV; int ldAV;
};


struct gpuHandler{

   curandGenerator_t   curandH;
   cusolverDnHandle_t  cusolverH;
   cublasHandle_t      cublasH;
   cusparseHandle_t    cusparseH;
};


struct jdqmr16Matrix {
	/* cpu matrix space */
   double  *values;
	int     *cols;
	int     *rows;

	/* gpu matrix space */
   double  *devValuesD;
 	float   *devValuesF;
 	half    *devValuesH;
	int     *devCols;
	int     *devRows;
   int     *devCsrRows;

	/* matrix general info */
	int dim;
	int nnz; 
};

struct jdqmr16Info {


   int numEvals =  1;
   int maxBasis =  15;
   int maxIter  =  1000;
   double tol      =  1e-04;
   double normMatrix = 0;
   struct jdqmr16Matrix  *matrix;
   struct devSolverSpace *sp;
   struct gpuHandler     *gpuH;

   struct initBasisSpace   *spInitBasis;
   struct eigHSpace        *spEigH;
   struct residualSpace    *spResidual;
   struct expandBasisSpace *spExpandBasis;
   struct restartSpace     *spRestart;
   struct innerSolverSpace *spInnerSolver;
   struct lockSpace        *spLock;

   int    numMatVecsfp64  = 0;
   int    numMatVecsfp16  = 0;
   int    outerIterations = 0;
   int    innerIterations = 0;

   int    useHalf = 1;
   int    locking = 1;
   double alpha   = 1; // is used in the inner loop for stopping 

   double   *gpuMemSpaceDouble;
   int      *gpuMemSpaceInt;
   void     *gpuMemSpaceVoid;

   int       gpuMemSpaceDoubleSize;
   int       gpuMemSpaceIntSize;
   size_t    gpuMemSpaceVoidSize;
};

void init_jdqmr16(struct jdqmr16Info *jd);
void destroy_jdqmr16(struct jdqmr16Info *jd);
void jdqmr16(struct jdqmr16Info *jd);
void jdqmr16_eigenpairs(double *V, int ldV, double *L, double *normr, struct jdqmr16Info *jd);

#endif


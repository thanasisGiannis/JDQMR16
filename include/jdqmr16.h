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
   
   double *Vprev; int ldVprev; // previous ritz vectors
   double *V;     int ldV;     // ritz vectors
   double *L;                  // ritz values
   double *W;     int ldW;     // space of projection (in the JD iteration)
   double *H;     int ldH;     // projected matrix
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
   void    *externalBufferTrans;

   double  *AV; int ldAV;
};

struct jdqmr16Info {
   int numEvals =  1;
   int maxBasis =  15;
   int maxIter  =  1000;
   int tol      =  1e-04;

   struct jdqmr16Matrix  *matrix;
   struct devSolverSpace *sp;
   struct gpuHandler     *gpuH;

   struct initBasisSpace *spInitBasis;
   struct eigHSpace      *spEigH;
};

void init_jdqmr16(struct jdqmr16Info *jd);
void destroy_jdqmr16(struct jdqmr16Info *jd);
void jdqmr16(struct jdqmr16Info *jd);


#endif


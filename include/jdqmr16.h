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


struct devSolverSpace{
   
   double *W; // space of projection (in the JD iteration)

};

struct jdqmr16Info {
   int numEvals = 1;

   int maxBasis = 15;
   int maxIter  = 1000;
   int tol      = 1e-04;

   struct jdqmr16Matrix* matrix;
   struct devSolverSpace *sp;
};

void init_jdqmr16(struct jdqmr16Info *jd);
void destroy_jdqmr16(struct jdqmr16Info *jd);
void jdqmr16();


#endif


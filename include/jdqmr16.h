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
   double  *hostValues;
	int     *hostColumn;
	int     *hostRow;

	/* gpu matrix space */
   double  *valuesD;
 	half    *valuesH;
	int     *column;
	int     *row;

	/* matrix general info */
	int dim;
	int nnz; 
};


struct jdqmr16Info {

   int dim      = NULL;
   int numEvals = 1;

   int maxBasis = 15;
   int maxIter  = 1000;
   int tol      = 1e-04;


   struct jdqmr16Matrix* mat;

};


#endif


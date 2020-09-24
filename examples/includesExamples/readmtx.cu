#include "readmtx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <unistd.h> 
#include <curand.h>
#include <cusolverDn.h>
#include <math.h>

#include <sys/time.h>



int readmtx(char *mName, struct matrix *A){
// Reading a file in coo format
	#include <stdio.h>
	#include "mmio.h"
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double *val;

     
    if ((f = fopen(mName, "r")) == NULL) 
         exit(1);
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);



	 int trueNNZ = 0;	
	 int tmpI,tmpJ;
	 double tmpVal;

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lf\n", &tmpI, &tmpJ, &tmpVal);

		  if(tmpI != tmpJ) trueNNZ++; 
    }

	 nz += trueNNZ;

    /* reseve memory for matrices */

	 //printf("M:%d N:%d nz: %d\n",M,N,nz); exit(0);
	
    A->rowIndex = (int *) malloc(nz * sizeof(int));
    A->colIndex = (int *) malloc(nz * sizeof(int));
    A->val = (double *) malloc(nz * sizeof(double));
	 A->diag = (double *)malloc(M*sizeof(double));
	 A->diagIndex = (int *)malloc(M*sizeof(int));
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
	
    val = A->val;
	 I   = A->rowIndex;
	 J   = A->colIndex;


	 int numData = nz - trueNNZ;
	 int j=0;

	 fseek(f, 0, SEEK_SET);
	 
	 int _M, _N, _nz;
    mm_read_mtx_crd_size(f, &_M, &_N, &_nz);
    
	 int diagCounter=0;
    for (i=0; i<numData; i++)
    {
        fscanf(f, "%d %d %lf\n", &I[j], &J[j], &val[j]);
        I[j]--;  /* adjust from 1-based to 0-based */
        J[j]--;
		  
		  if(I[j] != J[j]){
					val[j+1] = val[j];
					I[j+1] = J[j];
					J[j+1] = I[j];
					j++;
		  } 
			

		  if(I[j] == J[j]){
				A->diag[diagCounter] = 2048/(double)val[j];
			   A->diagIndex[diagCounter] = I[j];
				diagCounter++;
			}



		  j++;
    }


	 A->dim = M;
	 A->nnz = nz;



	quicksort(val ,0,A->nnz-1 ,I,J);

	 
	/* Allocate space for gpu */
	matrixStructMemAlloc(A);
	matrixSturctDevDataTransfer(A);

	return 0;
	
}

void quicksort(double *val ,int first,int last , int *I, int *J){
   int i, j, pivot, temp;
	double tmpVal;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(I[i]<=I[pivot]&&i<last)
            i++;
         while(I[j]>I[pivot])
            j--;
         if(i<j){


            tmpVal=val[i];
            val[i]=val[j];
            val[j]=tmpVal;


            temp=I[i];
            I[i]=I[j];
            I[j]=temp;



            temp=J[i];
            J[i]=J[j];
            J[j]=temp;


         }
      }

      tmpVal=val[pivot];
      val[pivot]=val[j];
      val[j]=tmpVal;


      temp=I[pivot];
      I[pivot]=I[j];
      I[j]=temp;


      temp=J[pivot];
      J[pivot]=J[j];
      J[j]=temp;

      quicksort(val,first,j-1, I,J);
      quicksort(val,j+1,last,I,J);

   }
}





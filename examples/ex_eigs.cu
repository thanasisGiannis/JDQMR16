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
#include "../include/jdqmr16.h"
#include "include/mmio.h"


void quicksort(double *val ,int first,int last , int *I, int *J);
int main(){

//	char *mName = "MTX_FILES/1138_bus.mtx"; //1138
//	char *mName = "MTX_FILES/msc04515.mtx"; //4,515
//	char *mName = "MTX_FILES/494_bus.mtx"; // 494
	char *mName = "MTX_FILES/nos4.mtx"; // 100
//	char *mName = "MTX_FILES/bcsstk01.mtx"; //48
//	char *mName = "MTX_FILES/finan512.mtx"; //74752
//	char *mName = "MTX_FILES/Andrews.mtx"; 
//	char *mName = "MTX_FILES/nd24k.mtx"; 
//	char *mName = "MTX_FILES/Lap7p1M.mtx"; 
//	char *mName = "MTX_FILES/G3_circuit.mtx"; // 1,585,478
	

   struct jdqmr16Matrix *A = (struct jdqmr16Matrix *)malloc(sizeof(struct jdqmr16Matrix));
   
   double *vA;     // values of matrix
   int    *rows;   // array of row indexing
   int    *cols;   // array of column indexing
   int    numRows; // number of rows
   int    numCols; // number of columns
   int    nnz;     // number of nonzero elements
   

   /* \/--- Loading Data ---\/ */
   int ret_code;
   MM_typecode matcode;
   FILE *f;



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
   if ((ret_code = mm_read_mtx_crd_size(f, &numRows, &numCols, &nnz)) !=0)
     exit(1);

   int trueNNZ = 0;
   int tmprows,tmpcols; 
   double tmpvA;
   for (int i=0; i<nnz; i++)
   {
     fscanf(f, "%d %d %lg\n", &tmprows, &tmpcols, &tmpvA);
     trueNNZ++;
     if(tmprows != tmpcols){
         trueNNZ++;
     }   
   }

   fseek(f, 0, 0);
   mm_read_banner(f, &matcode);
   mm_read_mtx_crd_size(f, &numRows, &numCols, &nnz);

   /* matrix memory allocation */ 
   rows = (int *) malloc(trueNNZ * sizeof(int));
   cols = (int *) malloc(trueNNZ * sizeof(int));
   vA   = (double *) malloc(trueNNZ * sizeof(double));


   int matIndex = 0;
   for (int i=0; i<nnz; i++)
   {
     fscanf(f, "%d %d %lg\n", &rows[matIndex], &cols[matIndex], &vA[matIndex]);
     rows[matIndex]--;  /* adjust from 1-based to 0-based */
     cols[matIndex]--;

     if(rows[matIndex] != cols[matIndex]){
         rows[matIndex+1] = cols[matIndex];
         cols[matIndex+1] = rows[matIndex];
         vA[matIndex+1]   = vA[matIndex];
      
         matIndex++;
     }   

     matIndex++;
   }
   nnz = trueNNZ;

   quicksort(vA ,0,nnz-1 ,rows,cols);
   if (f !=stdin) fclose(f);
   /* /\--- Done Loading Data ---/\ */


   /* Preparing data for jdqmr16 */
   A->values = vA;     // cpu values of matrix
   A->rows   = rows;   // array of row indexing
   A->cols   = cols;   // array of column indexing
   A->dim    = numRows; // or numCols (support for symmetric matrices)
   A->nnz    = nnz;     // number of nonzero elements
  

   struct jdqmr16Info* jd = (struct jdqmr16Info*)malloc(sizeof(struct jdqmr16Info));

   jd->numEvals = 10;     // number of wanted eigenvalues
   jd->maxBasis = 15;    // maximum size of JD basis
   jd->maxIter  = 1000;  // maximum number of JD iterations
   jd->tol      = 1e-04; // tolerance of the residual
   jd->matrix   = A;     // data of matrix


   init_jdqmr16(jd);
   jdqmr16(jd);
   destroy_jdqmr16(jd);


   
   /* free memory */
   free(vA);
   free(rows);
   free(cols);   
   free(A);

   free(jd);
	
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


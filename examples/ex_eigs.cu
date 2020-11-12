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
#include <cuda_profiler_api.h> 
#include <time.h>


void quicksort(double *val ,int first,int last , int *I, int *J);
int main(int argc,char *argv[]){


#if 0
	char *mName = "MTX_FILES/nos4.mtx"; // 100
//	char *mName = "MTX_FILES/1138_bus.mtx"; //1138
//	char *mName = "MTX_FILES/msc04515.mtx"; //4,515
//	char *mName = "MTX_FILES/494_bus.mtx"; // 494
// char *mName = "MTX_FILES/ex33.mtx";
// char *mName = "MTX_FILES/shallow_water1.mtx";
//	char *mName = "MTX_FILES/nd24k.mtx"; 
//	char *mName = "MTX_FILES/nasa4704.mtx"; 
	
   /* primme matrices */
//	char *mName = "MTX_FILES/finan512.mtx"; //74752
//	char *mName = "MTX_FILES/Andrews.mtx"; 
//	char *mName = "MTX_FILES/Lap7p1M.mtx"; 
//	char *mName = "MTX_FILES/cfd1.mtx"; 
//	char *mName = "MTX_FILES/cfd2.mtx"; 


//	char *mName = "MTX_FILES/nasasrb.mtx"; 
//	char *mName = "MTX_FILES/msdoor.mtx"; 
//	char *mName = "MTX_FILES/thermomech_dM.mtx"; 
//	char *mName = "MTX_FILES/G3_circuit.mtx"; 
//   char *mName = "MTX_FILES/nlpkkt160.mtx"; 
//   char *mName = "MTX_FILES/kkt_power.mtx"; 
//   char *mName = "MTX_FILES/Cube_Coup_dt6.mtx"; // half not working
//   char *mName = "MTX_FILES/CurlCurl_4.mtx"; 
//   char *mName = "MTX_FILES/Queen_4147.mtx"; 


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

#else

   struct jdqmr16Matrix *A = (struct jdqmr16Matrix *)malloc(sizeof(struct jdqmr16Matrix));
   
   double *vA;     // values of matrix
   int    *rows;   // array of row indexing
   int    *cols;   // array of column indexing
   int    numRows; // number of rows
   int    numCols; // number of columns
   int    nnz;     // number of nonzero elements
   int    n = 100;
   int    i,j;

   if(argc == 4){
      n = atoi(argv[3]);
   }

   nnz = n + 2*n-2;
   rows = (int*) calloc(nnz, sizeof(int));
   cols = (int*) calloc(nnz, sizeof(int));
   vA   = (double*) calloc(nnz, sizeof(double));

   int index = 0;
   for (i = 0; i < n; i++) {
      cols[index] = i;
      rows[index] = i;
      vA[index]   = 2.0;
      index++;

      if(i+1<n){
         cols[index] = i+1;
         rows[index] = i;
         vA[index]   = -1;
         index++;

         cols[index] = i;
         rows[index] = i+1;
         vA[index]   = -1;
         index++;

      }

   }
   numCols = n;
   numRows = n;
   quicksort(vA ,0,nnz-1 ,rows,cols);

#endif

   /* Preparing data for jdqmr16 */
   A->values = vA;      // cpu values of matrix
   A->rows   = rows;    // array of row indexing
   A->cols   = cols;    // array of column indexing
   A->dim    = numRows; // or numCols (support for symmetric matrices)
   A->nnz    = nnz;     // number of nonzero elements
  
   struct jdqmr16Info* jd = (struct jdqmr16Info*)malloc(sizeof(struct jdqmr16Info));

   if(argc > 1 ){
      jd->numEvals = atoi(argv[1]);          // number of wanted eigenvalues
      jd->maxBasis = 15;          // maximum size of JD basis
      jd->maxIter  = 3*numRows;;  // maximum number of JD iterations
      jd->tol      = 1e-7;       // tolerance of the residual
      jd->matrix   = A;           // data of matrix
      jd->useHalf  = atoi(argv[2]);
      jd->locking  = 0;
   }else{
      jd->numEvals = 1;          // number of wanted eigenvalues
      jd->maxBasis = 15;          // maximum size of JD basis
      jd->maxIter  = 3*numRows;;  // maximum number of JD iterations
      jd->tol      = 1e-7;       // tolerance of the residual
      jd->matrix   = A;           // data of matrix
      jd->useHalf  = -2;
      jd->locking  = 0;
   }
   init_jdqmr16(jd);

   printf("%%Finding eigenpairs...\n");

   time_t start = time(NULL);
   cudaProfilerStart();
   jdqmr16(jd);
   cudaProfilerStop();
   time_t end = time(NULL);
   printf("%%Found them!\n");

   double *V = (double*)malloc(sizeof(double)*(A->dim)*(jd->numEvals));
   double *L = (double*)malloc(sizeof(double)*(jd->numEvals));
   double *normr = (double*)malloc(sizeof(double)*(jd->numEvals));
   jdqmr16_eigenpairs(V,A->dim,L, normr, jd);


   destroy_jdqmr16(jd);

   #if 1
   printf("\n\n%%==========\n");
   printf("\n\n%%-- EigenValues --\n");
   for(int i=0;i<(jd->numEvals);i++){
      printf("%%L[%d]=%e; normr[%d]/normA = %e; \n",i,L[i],i,normr[i]/jd->normMatrix);
   }
   printf("\n\n%%----------\n");
   printf("%%outerIterations=%d \n%%innerIterations=%d\n%%Tolerance=%e\n%%normA=%e\n",jd->outerIterations,jd->innerIterations,jd->tol,jd->normMatrix);
   printf("%%dim=%d;\n",numRows);
   printf("%%nnz=%d;\n",nnz);
   printf("%%nnz(%%)=%e %%;\n",(float)nnz/((float)numRows*numRows));
   printf("%%fp64 matVecs=%d\n%%fp16 matVecs=%d\n",jd->numMatVecsfp64,jd->numMatVecsfp16);
   printf("\n\n%%----------\n");
   printf("%%==========\n");
   printf("%%Wall clock time: %f seconds\n", difftime(end, start));
   #endif         




   /* free memory */
   free(L);
   free(V);
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

   if(first >= last)
      return;

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
         }else{
            break;
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

      if(first <= j-1){
         quicksort(val,first,j-1, I,J);
      }
      if(j+1 <= last){
         quicksort(val,j+1,last,I,J);
      }
   }
}


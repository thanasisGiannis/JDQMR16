#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include "../matrix/double2halfMat.h"
#include <cublas.h>
#include "../include/helper.h"

void printMatrixHalf(half *matrix16, int rows, int cols, char *name){

	 double *hmatrix = (double*)malloc(sizeof(double)*rows*cols);
    double *matrix ; cudaMalloc((void**)&matrix,sizeof(double)*rows*cols);

//return;
    half2doubleMat(matrix, rows, matrix16, rows, rows, cols);

	 cudaMemcpy(hmatrix,matrix,sizeof(double)*rows*cols,cudaMemcpyDeviceToHost);

	 printf("%% %s\n",name);
	 printf("%% ===================\n");

    printf("%s = zeros(%d,%d);\n",name,rows,cols);
    for(int row = 0 ; row < rows ; row++){
        for(int col = 0 ; col < cols ; col++){
            double Areg = hmatrix[row + col*rows];
//            printf("%s(%d,%d) = %.50lf;\n", name, row+1, col+1, Areg);
            printf("%s(%d,%d) = %e;\n", name, row+1, col+1, Areg);
        }
    }


	printf("%% ===================\n\n");
	free(hmatrix);
   cudaFree(matrix);
}



void printMatrixDouble(double *matrix, int rows, int cols, char *name){

	double *hmatrix = (double*)malloc(sizeof(double)*rows*cols);
	cudaMemcpy(hmatrix,matrix,sizeof(double)*rows*cols,cudaMemcpyDeviceToHost);

	printf("%% %s\n",name);
	printf("%% ===================\n");

	 printf("%s = zeros(%d,%d);\n",name,rows,cols);
    for(int row = 0 ; row < rows ; row++){
        for(int col = 0 ; col < cols ; col++){
            double Areg = hmatrix[row + col*rows];
//            printf("%s(%d,%d) = %.50lf;\n", name, row+1, col+1, Areg);
            printf("%s(%d,%d) = %e;\n", name, row+1, col+1, Areg);
        }
    }


	printf("%% ===================\n\n");
	free(hmatrix);
}

void printMatrixInt(int *matrix, int rows, int cols, char *name){

	int *hmatrix = (int*)malloc(sizeof(int)*rows*cols);
	cudaMemcpy(hmatrix,matrix,sizeof(int)*rows*cols,cudaMemcpyDeviceToHost);

	printf("%% %s\n",name);
	printf("\n\n %% ===================\n");

	 printf("%s = zeros(%d,%d);\n",name,rows,cols);
    for(int row = 0 ; row < rows ; row++){
        for(int col = 0 ; col < cols ; col++){
            int Areg = hmatrix[row + col*rows];
            printf("%s(%d,%d) = %d;\n", name, row+1, col+1, Areg);
        }
    }


	printf("\n\n %% ===================\n");
	free(hmatrix);

}





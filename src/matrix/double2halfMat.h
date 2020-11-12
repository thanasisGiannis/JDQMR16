#ifndef DOUBLE2HALFMAT_H
#define DOUBLE2HALFMAT_H

cudaError_t double2halfMat(half *A16, int ldA16, double *A, int ldA, int rows, int cols);
cudaError_t half2doubleMat(double *A, int ldA, half *A16, int ldA16, int rows, int cols);
cudaError_t float2doubleMat(double *A, int ldA, float *A32, int ldA32, int rows, int cols);
cudaError_t double2floatMat(float *A32, int ldA32, double *A, int ldA, int rows, int cols);
#endif

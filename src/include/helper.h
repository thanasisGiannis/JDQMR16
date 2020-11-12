#ifndef HELPER_H
#define HELPER_H

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
  _a < _b ? _a : _b; })


static const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

#define CUBLAS_CALL(call)  \
        do {\
            cublasStatus_t err = call;\
            if (CUBLAS_STATUS_SUCCESS != err) \
            {\
                printf( "CUBLAS error in %s (%d): %s\n", __FILE__ , __LINE__ ,cublasGetErrorString(err));\
                exit(EXIT_FAILURE);\
            }\
			cudaDeviceSynchronize();\
        } while(0)

#define CUDA_CALL(call)  \
        do {\
            cudaError_t err = call;\
            if (cudaSuccess != err) \
            {\
                printf( "CUDA error in %s (%d): %s\n", __FILE__ , __LINE__ ,cudaGetErrorString(err));\
                exit(EXIT_FAILURE);\
            }\
			cudaDeviceSynchronize();\
        } while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}cudaDeviceSynchronize();} while(0)

/*
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}cudaDeviceSynchronize();} while(0)
*/
#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}cudaDeviceSynchronize();} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(0);}cudaDeviceSynchronize();} while(0)



#if 1
#include <cuda_fp16.h>
void printMatrixHalf(half *matrix16, int rows, int cols, char *name);
void printMatrixDouble(double *matrix, int rows, int cols, char *name);
void printMatrixInt(int *matrix, int rows, int cols, char *name);
void printMatrixFloat(float *matrix, int rows, int cols, char *name);

#endif

#endif

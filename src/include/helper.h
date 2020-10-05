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
    return EXIT_FAILURE;}cudaDeviceSynchronize();} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}cudaDeviceSynchronize();} while(0)

#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}cudaDeviceSynchronize();} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}cudaDeviceSynchronize();} while(0)



#if 1
void printMatrixDouble(double *matrix, int rows, int cols, char *name);
void printMatrixInt(int *matrix, int rows, int cols, char *name);
#endif

#endif

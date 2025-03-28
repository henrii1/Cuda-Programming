/*
Common utilities for CUDA code
*/

#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cublasLt.h>


// cuBLAS Precision settings

#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else
#define CUBLAS_LOWP CUDA_R_16BF
#endif


// global settings for cublas
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// checking for errors
inline void cublasCheck(cublasStatus_t status, const char* file, int line){
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("[CUBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}

#define cublasCheck(err) { cublasCheck(err, __FILE__, __LINE__);}


#endif // CUBLAS_COMMON_H
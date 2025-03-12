// common utils for cuda code

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>   //std::bool_constant
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utils.h"

//Global defines and settings

extern cudaDeviceProp deviceProp;

#define WARP_SIZE 32U

#if __CUDA_ARCH__ == 800 || __CUDA_ARXH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// convenient architecture for calculating grid/block dim for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// compiler time boolean value
constexpr std::bool_constant<true> True;  // True.value == true
constexpr std::bool_constant<false> False;  // std::bool_constant is a data type


/* Error checking*/
 inline void cudaCheck_(cudaError_t error, const char *file, int line){
    if (error != cudaSuccess){
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 }
 #define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

 // updated version to cudafree with error checking
 template<class T>
 inline void cudaFreeCheck(T** ptr, const char* file, int line){
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess){
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
 }
 #define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

/* CUDA Precision settings*/

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific precison based configurations
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
#elif defined(ENABLE_FP16)
typedef half floatX;
#define PRECISION_MODE PRECISION_FP16
#else  // Defaults to BF16
typedef __nv_bfloat16 floatX;
#define PRECISION_MODE RECISION_BF16
#endif


/* Load and store with streaming cache hints */

#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX* address){
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}sample_softmax

__device__ void __strcs(floatX* address, floatX value){
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}

#endif

/* Profiler utils */

class NvtxRange {
    public:
    NvtxRange(const char* s) {nvtxRangePush(s); }
    NvtxRange(const std::string& base_str, int number){
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    ~NvtxRange(){ nvtxRangePop(); }
}
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)


/* Utilities to Read & Write between CUDA memory to and from files*/

inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream){
    // allocate pinned buffer for faster, async transfer
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);

    while(rest_bytes > 0){
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is ongoing, transfer write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));   // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}


// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
    // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
   char* buffer_space;
   cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
   // split allocation in two
   void* read_buffer = buffer_space;
   void* write_buffer = buffer_space + buffer_size;

   // prime the read buffer;
   char* gpu_write_ptr = (char*)dest;
   size_t copy_amount = std::min(buffer_size, num_bytes);
   freadCheck(read_buffer, 1, copy_amount, src);

   size_t rest_bytes = num_bytes - copy_amount;
   size_t write_buffer_size = copy_amount;
   std::swap(read_buffer, write_buffer);

   // now the main loop; as long as there are bytes left
   while(rest_bytes > 0) {
       // initiate next read
       copy_amount = std::min(buffer_size, rest_bytes);
       cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
       gpu_write_ptr += write_buffer_size;
       // while this is going on, read from disk
       freadCheck(read_buffer, 1, copy_amount, src);
       cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

       std::swap(read_buffer, write_buffer);
       rest_bytes -= copy_amount;
       write_buffer_size = copy_amount;
   }

   // copy the last remaining write buffer to gpu
   cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
   cudaCheck(cudaStreamSynchronize(stream));
   cudaCheck(cudaFreeHost(buffer_space));
}



#endif
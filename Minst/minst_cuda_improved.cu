#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 4


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA Error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void init_random(float *data, int size, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state);
    }
}

__global__ void relu_derivative(float *data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        data[idx] *= (data[idx] > 0);
    }
}

__global__ void backward_pass_naive(float *input, float *hidden, float *output, int *labels,
    float *weights1, float *weights2,
    float *grad_weights1, float *grad_weights2,
    float *grad_bias1, float *grad_bias2,
    int input_size, int hidden_size, int output_size, int batch_size){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch_idx = blockIdx.y;

        __shared__ grad_output[OUTPUT_SIZE];

        if (idx < output_size && batch_idx < batch_size){
            grad_output[idx] = output[batch_idx * output_size + idx];
            if (idx == labels[batch_idx]){
                grad_output[idx] -= 1.0f;
            }
        }
        __syncthreads();

        //next part, hidden size and batch size
        if (idx < hidden_size && batch_idx < batch_size){
           grad_hidden = 0.0f;
           for (int i=0; i<output_size; i++){
            grad_hidden += grad_output[i] * weights2[i * hidden_size + idx];  //dY/DO @ W2.T
           } 
           grad_hidden *= (hidden[batch_idx * hidden_size + idx] > 0)

           for (int i = 0; i < input_size; i++) {
            atomicAdd(&grad_weights1[idx * input_size + i], grad_hidden * input[batch_idx * input_size + i]);
        }
        atomicAdd(&grad_bias1[idx], grad_hidden);

        }
        if (idx < output_size * hidden_size && batch_idx < batch_size) {
            int i = idx / hidden_size;
            int j = idx % hidden_size;
            atomicAdd(&grad_weights2[idx], grad_output[i] * hidden[batch_idx * hidden_size + j]);
        }
    
        if (idx < output_size && batch_idx < batch_size) {
            atomicAdd(&grad_bias2[idx], grad_output[idx]);
    }
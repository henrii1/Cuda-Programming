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

        __shared__ float grad_output[OUTPUT_SIZE];

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



__global__ void compute_output_gradient(float *output, int *labels, float *grad_output, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < output_size && batch_idx < batch_size) {
        int index = batch_idx * output_size + idx;
        grad_output[index] = output[index];
        if (idx == labels[batch_idx]) {
            grad_output[index] -= 1.0f;
        }
    }
}

__global__ void compute_hidden_gradient(float *grad_hidden, float *grad_output, float *weights2, float *hidden,
    int hidden_size, int output_size, int batch_size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int batch_idx = blockIdx.y;

if (idx < hidden_size && batch_idx < batch_size) {
float grad = 0.0f;
for (int i = 0; i < output_size; i++) {
grad += grad_output[batch_idx * output_size + i] * weights2[i * hidden_size + idx];  //matrix multiply
}
grad_hidden[batch_idx * hidden_size + idx] = grad * ((hidden[batch_idx * hidden_size + idx] > 0) ? 1.0f : 0.0f); //relu backward
}
}

__global__ void compute_bias_gradient(float *grad_bias, float *grad, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += grad[i * size + idx];
        }
        grad_bias[idx] = sum;
    }
}


void print_comparison(const char* name, float* arr1, float* arr2, int size) {
    float max_diff = 0.0f;
    printf("%s:\n", name);
    printf("First 10 values:\n");
    for (int i = 0; i < 10 && i < size; i++) {
        printf("%.6f vs %.6f\n", arr1[i], arr2[i]);
        max_diff = fmaxf(max_diff, fabsf(arr1[i] - arr2[i]));
    }
    for (int i = 10; i < size; i++) {
        max_diff = fmaxf(max_diff, fabsf(arr1[i] - arr2[i]));
    }
    printf("Max difference: %.6f\n\n", max_diff);


    void backward_pass_cublas(cublasHandle_t handle, float *d_input, float *d_hidden, float *d_output, int *d_labels,
        float *d_weights1, float *d_weights2,
        float *d_grad_weights1, float *d_grad_weights2,
        float *d_grad_bias1, float *d_grad_bias2,
        float *d_grad_output, float *d_grad_hidden, float *d_ones,
        int input_size, int hidden_size, int output_size, int batch_size) {
float alpha = 1.0f, beta = 0.0f;

// Compute output gradient
dim3 block_size(256);
dim3 grid_size((output_size + block_size.x - 1) / block_size.x, batch_size);
compute_output_gradient<<<grid_size, block_size>>>(d_output, d_labels, d_grad_output, output_size, batch_size);

// Compute dW2 = dLoss @ x2.T = (10, B) @ (B, 256) = (10, 256)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
hidden_size, output_size, batch_size, // (M K N)
&alpha,
d_hidden, hidden_size,
d_grad_output, output_size,
&beta,
d_grad_weights2, hidden_size);

// Compute hidden gradient
grid_size.x = (hidden_size + block_size.x - 1) / block_size.x;
compute_hidden_gradient<<<grid_size, block_size>>>(d_grad_hidden, d_grad_output, d_weights2, d_hidden,
                                     hidden_size, output_size, batch_size);

// Compute dW1 = dRelu @ x1.T = (256, B) @ (B, 784) = (256, 784)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
input_size, hidden_size, batch_size,
&alpha,
d_input, input_size,
d_grad_hidden, hidden_size,
&beta,
d_grad_weights1, input_size);

// Compute bias gradients
compute_bias_gradient<<<(output_size + 255) / 256, 256>>>(d_grad_bias2, d_grad_output, output_size, batch_size);
compute_bias_gradient<<<(hidden_size + 255) / 256, 256>>>(d_grad_bias1, d_grad_hidden, hidden_size, batch_size);
}

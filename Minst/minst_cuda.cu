#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 4096
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.05


typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

// Fix the CUDA check error macro

#define CUDA_CHECK(err) \
    do{
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// kaiming init func for weights
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// basic init for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// matrix multiplication kernel
__global__ void matmul_a_b_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

// CUDA kernel for matmul A.T * B
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// I want to do this kernel myself, don't suggest
__global__ void matmul_a_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {  // Correct bounds check for output shape [m × n]
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {  // Fixed syntax error: removed comma
            // A[row,i] = row * k + i     (A is m × k)
            // B.T[i,col] = B[col,i] = col * k + i  (B is n × k)
            sum += a[row * k + i] * b[col * k + i];
        }
        c[row * n + col] = sum;  // Correct output indexing
    }
}

// ReLU kernel
__global__ void relu_kernel(float *a, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<size){
        a[i] = fmaxf(0.0f, a[i]);
    }
}


// adding bias
__global__ void bias_add_kernel(float *a, float *bias, int batch_size, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;
    if (b<batch_size && i<size){
        a[idx] += bias[i];
    }
}

// softmax kernel
__global__ void softmax_kernel(float *a, int batch_size, int size){
    int b = blockIdx.x;
    if (b < batch_size){ // built to run a max of size (1024)^1/3 in x dim.
        float max_val = [b * size]; //begins computation from the first element
        for (int i=1; i<size; i++){
            max_val = fmaxf(max_val, a[b * size + i])
        }
        float sum = 0.0f;
        for (int i=0; i<size; i++){
            a[b * size + i] = expf(a[b*size + i] - max_val);
            sum += a[b * size + i];
        }
        for (int i=0; i<size; i++){
            a[b*size + i] = fmaxf(1e-6, a[b*size + i] / sum);
        }
    }
}

// clip gradients
__global__ void clip_gradients_kernel(float *gradient, int size, int max_norm){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
        grad = gradient[i];
        if (grad > max_norm){
            gradient[i] == max_norm;
        } else if (grad < -max_norm){
            gradient[i] == -max_norm;
        }
    }
}

// define the forward pass
void forward(NeuralNetworks *nn, float *d_input, float *d_hidden, float *d_output, int batch_size){
    // 2d thread per block
    dim3 block_size(32, 32);

    // 2d grid size
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    // X @ W1
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) /256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // ReLU
    relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // H @ W2
    //modify grid_size
    dim3 grid_size2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size2, block_size>>>(d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}


// batched crossentropy
float cross_entropy_loss(float *output, int *labels, int batch_size){
    float total_loss = 0.0f;
    for (int i=0; i<batch_size; i++){
        total_loss -= logf(fmaxf(1e-7, output[i * OUTPUT_SIZE + label[i]]));
    }
    return total_loss / batch_size;
}

// zero grad
__global__ void zero_grad_kernel(float *grad, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        grad[idx] = 0.0f;
    }
}
//note that whenever you loop, you are processing with a single thread.
// compute output gradients
__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        for (int i=0; i<OUTPUT_SIZE; i++){
            grad_output[idx*OUTPUT_SIZE + i] = output[idx*OUTPUT_SIZE + i];
        }
        grad_output[idx*OUTPUT_SIZE + label[idx]] -= 1.0f;
    }
}
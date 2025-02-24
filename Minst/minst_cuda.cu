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

//for matrix multiplication, we need 3 identifiers to index into A, B, and C. this would be done using their shapes.
// Think of matrix multiplications as shapes.
// update the gradients
// dW = (1/batch_size) * dY.T @ X (Linear Layer)
//dW = (1/batch_size) * dY.T @ X
__global__ void update_gradients_kernel(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        //This is just a basic matmul. atomic add is used in the case as a precaution when gradients sum up, it does current_value + new_value
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

        if (j == 0) {
            float grad_b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];
            }
            atomicAdd(&grad_bias[i], grad_b_sum);   //shape is curr_size
        }
    }
}


// dRelu
__global__ void drelu_kernel(float *x, float *d_ReLU_out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        d_ReLU_out[i] = (x[i] > 0.0f);
    }
}

// Elementwise multiply of d_dX2 and d_grad_hidden

__global__ void multiply_gradients_kernel(float *grad1, float *grad2, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        grad1[idx] *= grad2[idx];
    }
}

/* General Approach to NN with C, CPP and CUDA:
1. Build a struct to contain all weights and biases
2. Write all forward and backward functions
4. Create a forward function that joins all forward related functions, weights, bias and activation memory
5. Create a backward function that joins all backward related functions, dweights, dbias and dactivation memory
6. Write the update functions and put them together in a single update function.
7. write the train function the joins the forward, backward and update functions to train the model.
8. Within the int main function assign memory, load input data and train the model

*/

// backward pass
void backward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels, int batch_size) {
    // 1. Zero all gradient buffers
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    zero_grad_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    zero_grad_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    // 2. Compute the output gradients (d_grad_output)
    float *d_grad_output;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    // Here compute_output_gradients_kernel writes d_grad_output = d_output, then subtracts 1 at the correct label index.
    compute_output_gradients_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_grad_output, d_output, d_labels, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    // 3. Update gradients for layer 2 (W2 and b2)
    // dW2 = d_grad_output.T @ d_hidden   ;   db2 = sum(d_grad_output, axis=0)
    // Note: Since our kernel uses blockIdx.y for the output neuron (i) and threads along x for the input neuron (j),
    // we want one block per output neuron. Therefore, use block dim (32,1) and grid.y = OUTPUT_SIZE.
    {
        dim3 block_size(32, 1);
        dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, OUTPUT_SIZE);
        update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 4. Backpropagate through layer 2 to compute d_grad_hidden:
    // d_grad_hidden = d_grad_output * (W2)ᵀ.
    float *d_grad_hidden;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_SIZE * sizeof(float)));
    {
        // matmul_a_bt_kernel computes: c = A * (B)ᵀ,
        // with A = d_grad_output [batch_size x OUTPUT_SIZE],
        // and B = nn->weights2 [HIDDEN_SIZE x OUTPUT_SIZE].
        // Set m = batch_size, n = HIDDEN_SIZE, k = OUTPUT_SIZE.
        dim3 block_size(32, 32);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x, (HIDDEN_SIZE + block_size.y - 1) / block_size.y);
        matmul_a_bt_kernel<<<grid_size, block_size>>>(d_grad_output, nn->weights2, d_grad_hidden, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 5. Account for ReLU derivative: d_grad_hidden = d_grad_hidden ⊙ (d_hidden > 0)
    float *d_grad_relu;
    CUDA_CHECK(cudaMalloc(&d_grad_relu, batch_size * HIDDEN_SIZE * sizeof(float)));
    {
        int total_hidden = batch_size * HIDDEN_SIZE;
        int blockDim = 256;
        int gridDim = (total_hidden + blockDim - 1) / blockDim;
        // drelu_kernel computes d_ReLU_out as 1 where d_hidden > 0, else 0.
        drelu_kernel<<<gridDim, blockDim>>>(d_hidden, d_grad_relu, total_hidden);
        CUDA_CHECK(cudaGetLastError());
        
        // Multiply elementwise: d_grad_hidden *= d_grad_relu
        multiply_gradients_kernel<<<gridDim, blockDim>>>(d_grad_hidden, d_grad_relu, total_hidden);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 6. Update gradients for layer 1 (W1 and b1)
    // dW1 = d_grad_hidden.T @ d_input   ;   db1 = sum(d_grad_hidden, axis=0)
    {
        // Now d_grad_hidden is [batch_size x HIDDEN_SIZE] and d_input is [batch_size x INPUT_SIZE].
        // Use update_gradients_kernel with: prev_layer = d_input, curr_size = HIDDEN_SIZE, prev_size = INPUT_SIZE.
        dim3 block_size(32, 1);
        dim3 grid_size((INPUT_SIZE + block_size.x - 1) / block_size.x, HIDDEN_SIZE);
        update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights1, nn->grad_bias1, d_grad_hidden, d_input, batch_size, INPUT_SIZE, HIDDEN_SIZE);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 7. Cleanup temporary buffers
    cudaFree(d_grad_output);
    cudaFree(d_grad_hidden);
    cudaFree(d_grad_relu);
}


// Modify evaluate_accuracy to handle larger datasets by processing in batches
float evaluate_accuracy(NeuralNetwork *nn, float *d_X_test, int *d_y_test, float *d_hidden, float *d_output, int total_size) {
    int num_batches = (total_size + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_correct = 0;
    int total_processed = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int current_batch_size = (batch == num_batches - 1) ? 
            (total_size - batch * BATCH_SIZE) : BATCH_SIZE;
        
        if (current_batch_size <= 0) break;

        forward(nn, &d_X_test[batch * BATCH_SIZE * INPUT_SIZE], 
                d_hidden, d_output, current_batch_size);
        
        float *h_output = (float *)malloc(current_batch_size * OUTPUT_SIZE * sizeof(float));
        int *h_y_test = (int *)malloc(current_batch_size * sizeof(int));
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, 
            current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_test, &d_y_test[batch * BATCH_SIZE], 
            current_batch_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < current_batch_size; i++) {
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (h_output[i * OUTPUT_SIZE + j] > h_output[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }
            if (predicted == h_y_test[i]) {
                total_correct++;
            }
        }
        
        total_processed += current_batch_size;
        free(h_output);
        free(h_y_test);
    }
    
    return 100.0f * total_correct / total_processed;
}


// Modify train function
void train(NeuralNetwork *nn, float *X_train, int *y_train, float *X_test, int *y_test) {
    float *d_X_train, *d_X_test, *d_hidden, *d_output;
    int *d_y_train, *d_y_test;

    // Allocate memory for training and test data
    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_test, TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y_test, TEST_SIZE * sizeof(int)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test, TEST_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        // Zero out gradients at the beginning of each epoch
        zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
        zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
        zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
        zero_grad_kernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            forward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, BATCH_SIZE);

            float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            float loss = cross_entropy_loss(h_output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            free(h_output);

            backward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, &d_y_train[start_idx], BATCH_SIZE);
            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                // Use random batch from test set for accuracy reporting
                int test_start_idx = rand() % (TEST_SIZE - BATCH_SIZE);
                float test_accuracy = evaluate_accuracy(nn, 
                    &d_X_test[test_start_idx * INPUT_SIZE],
                    &d_y_test[test_start_idx],
                    d_hidden, d_output, BATCH_SIZE);
                
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Test Accuracy: %.2f%%\n", 
                       epoch + 1, EPOCHS, batch + 1, num_batches, 
                       total_loss / (batch + 1), test_accuracy);
            }
        }
        
        // Evaluate on entire test set at end of epoch
        float test_accuracy = evaluate_accuracy(nn, d_X_test, d_y_test, d_hidden, d_output, TEST_SIZE);
        printf("Epoch %d/%d completed, Loss: %.4f, Test Accuracy: %.2f%%\n", 
            epoch + 1, EPOCHS, total_loss / num_batches, test_accuracy);
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_train));
    CUDA_CHECK(cudaFree(d_y_test));
}



// Modified initialize function to allocate memory for gradients
void initialize_neural_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    // Allocate temporary host memory
    float *h_weights1 = (float *)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases on the host
    initialize_weights(h_weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(h_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    // Copy initialized values to device
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Free temporary host memory
    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}



int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    load_data("../../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../../mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("../../mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../../mnist_data/y_test.bin", y_test, TEST_SIZE);

    // print first image in the terminal
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f) {
                printf("X");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("First 10 training labels: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", y_train[i]);
    }
    printf("\n");
    
    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    train(&nn, X_train, y_train, X_test, y_test);

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate duration in seconds with milliseconds
    double training_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("\nTotal training time: %.2f sec\n", training_time);

    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.grad_weights1));
    CUDA_CHECK(cudaFree(nn.grad_weights2));
    CUDA_CHECK(cudaFree(nn.grad_bias1));
    CUDA_CHECK(cudaFree(nn.grad_bias2));
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}



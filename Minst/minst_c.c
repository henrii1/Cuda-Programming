#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define EPOCH 10
#define BATCH_SIZE 4
#define LEARNING_RATE 0.01


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

// Load batched img data
void load_data((const char *filename, float *data, int size){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size){
        fprintf(stderr, "Error: Unable to read file. Expected %d, but got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size){
        fprintf(stderr, "Error: Unable to read file. Expected %d, but got %zu\n", size, read_size);
        exit(1);
    }
}

// Kaiming init for weights
void initialize_weights(float *weights, int size){
    float scale = sqrtf(2.0f / size);
    for (i=0; i<size; i++){
        weights[i]  = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// Initialize bias
void initialize_bias(float *bias, int size){
    for (i=0; i<size; i++){
        bias[i] = 0.0f;
    }
}


// Modify softmax to work with batches
void softmax(float *x int num_batches, int size){
    for (int b=0; b<num_batches; b++){
        float max = x[b * size];
        for (int i=1; i<size; i++){
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f;
        for (int i=0; i<size; i++){
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }

        for (int i=0; i<size; i++){
            x[b * size + i] = fmaxf(1e-7f, x[b * size + i] / sum);
        }
    }
}

// matmul  a@b
void matmul_a_b(float *A, float *B, float *C, int m, int n, int k){
    for (int i=0; i<m; i++){
        for (int j=0; j<k; j++){
            C[i * k + j] = 0.0f;

            for (int l=0; l<n; l++){
                c[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}


// Matrix multiplication A@B.T
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k){
    for (int i=0; i<m; i++){
        for (int j=0; j<k; j++){
            C[i * k + j] = 0.0f;

            for (int l=0; l<n; l++){
                c[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}


// Matmul A.T@B
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k){
    for (int i=0; i<n; i++){
        for (int j=0; j<k; j++){
            C[i * k + j] = 0.0f;

            for (int l=0; l<m; l++){
                c[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

// ReLU forward
void relu_forward(float *X, int size){
    for (int i=0; i<size; i++){
        X[i] = fmaxf(0.0f, x[i]);
    }
}


// Add Bias forward
void bias_forward(float *X, float *bias, int num_batches, int size){
    for (int b=0; b<num_batches; b++){
        for (int s=0; s<size; s++){
            X[b * size + s] += bias[s];
        }
    }
}


// Modified forward function

void forward(NeuralNetwork *nn, float *input, float *hidden, float *output, int num_batches){
    matmul_a_b(intput nn->weights1, hidden, num_batches, INPUT_SIZE, HIDDEN_SIZE);
    bias_forward(hidden, nn->bias1, num_batches, HIDDEN_SIZE);
    relu_forward(hidden, num_batches * HIDDEN_SIZE);

    matmul_a_b(hidden, nn->weights2, output, num_batches, HIDDEN_SIZE, OUTPUT_SIZE);
    bias_forward(output, nn->bias2, num_batches, OUTPUT_SIZE);
    softmax(output, num_batches, OUTPUT_SIZE);
}

// Modify cross entropy to work with batches
float cross_entropy_loss(float *output, int *labels, int num_batches){
    float loss = 0.0f;
    for (int b=0; b<num_batches; b++){
        total_loss += -logf(fmaxf(1e-7f, output[b * OUTPUT_SIZE + labels[b]]));
    }
    total_loss /= num_batches;
    
}

// Zero out gradients
void zero_grad(float *grad, int size){
    memset(grad, 0, sizeof(float) * size);
}

// ReLU backward
void relu_backward(float *grad, float *X, int size){
    for (int i=0; i<size; i++){
        grad[i] *= (X[i] > 0.0f);
    }
}


// Bias backward
void bias_backward(float *grad_bias, float *grad, int num_batches, int size){
        for (int i=0; i<size; i++){
            grad_bias[i] = 0.0f;
            for (int b=0; b<num_batches; b++){
                grad_bias[i] += grad[b * size + i];
            }
        }
    }

// compute gradients for output layer
void compute_output_gradients(float *grad_output, float *output, int *labels, int num_batches){
    for (int b=0; b<num_batches; b++){
        for (int i=0; i<OUTPUT_SIZE; i++){
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];

        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;         //Reduce the gradient of the right idx
    }
}

// update gradients for weights and biases

void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int num_batches, int prev_size, int cur_size){
    for (int i=0; i<cur_size; i++){
        for (int j=0; j<prev_size; j++){
            for (int b=0; b<num_batches; b++){
                grad_weights[i * prev_size + j] += grad_layer[b * cur_size + i] * prev_layer[b * prev_size + j];
            }
        }
        for (int b=0; b<num_batches; b++){
            grad_bias[i] += grad_layer[b * cur_size + i];
        }
    }
}



// Complete the backward pass
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int num_batches){

    // initialize grads to zero
    zero_grad(nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // compute grads for output
    float *grad_output = (float *)malloc(num_batches * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, num_batches);

    // update grads for weight2 layer == grad_output.T @ hidden
    matmul_at_b(hidden, grad_output, nn->grad_weights2, num_batches, HIDDEN_SIZE, OUTPUT_SIZE);

    // update grads for bias2 layer
    bias_backward(nn->grad_bias2, grad_output, num_batches, OUTPUT_SIZE);

    // update grad for hidden layer  == grad_output @ weights2.T
    float *dX2 = (float *)malloc(num_batches * HIDDEN_SIZE * sizeof(float));
    matmul_a_bt(grad_output, nn->weights2, dX2, num_batches, OUTPUT_SIZE, HIDDEN_SIZE);

    // relu
    float *d_ReLU_out = (float *)malloc(num_batches * HIDDEN_SIZE * sizeof(float));
    relu_backward(d_ReLU_out, dX2, num_batches*HIDDEN_SIZE);

    //update grads for weight1
    matmul_at_b(input, d_ReLU_out, nn->grad_weights1, num_batches, INPUT_SIZE, HIDDEN_SIZE);
    bias_backward(nn->grad_bias1, d_ReLU_out, num_batches, HIDDEN_SIZE);

    free allocated memory
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}


// updating the gradient step
void update_weights(NeuralNetwork *nn){
    for (int i=0; i< HIDDEN_SIZE * INPUT_SIZE; i++){
        nn->weights1[i] -= LEARNING_RATE * nn->grad_weights1[i];

    }
    for (int i=0; i< OUTPUT_SIZE * HIDDEN_SIZE; i++){
        nn->weights2[i] -= LEARNING_RATE * nn->grad_weights2[i];

    }
    for (int i=0; i<HIDDEN_SIZE; i++){
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }
    for (int i=0; i<OUTPUT_SIZE; i++){
        nn->bias2[i] -= LEARNING_RATE * nn->bias2[i];
    }
}


//Modify train function for batches
void train(NeuralNetwork *nn, float *X_train, int *y_train){
    float *hidden = (float *)malloc(BATCH_SIZE*HIDDEN_SIZE*sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE*OUTPUT_SIZE*sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch=0; epoch<EPOCH; epoch++){
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch=0; batch<num_batches; batch++){
            int start_idx = batch * BATCH_SIZE;

            forward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, BATCH_SIZE);

            float loss = cross_entropy_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            for (int i=0; i<BATCH_SIZE; i++){
                int predicted_idx = 0;
                for (int j=1; j<OUTPUT_SIZE; j++){
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted_idx]){
                        predicted_idx = j
                    }
                }
                    if (predicted_idx == y_train[start_idx + i]) correct++;
                }

            backward(nn, &X_train[start_idx*INPUT_SIZE], hidden, output, &y_train[start_idx], BATCH_SIZE);
            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", 
                       epoch + 1, EPOCH, batch + 1, num_batches, total_loss / (batch + 1), 
                       100.0f * correct / ((batch + 1) * BATCH_SIZE));
            }

            }
            printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", 
                epoch + 1, EPOCH, total_loss / num_batches, 100.0f * correct / TRAIN_SIZE); 
        }
        free(hidden);
        free(output);
}

// Modify the initialize function to allocate memory for gradients
void initialize_neural_network(NeuralNetwork *nn) {
    nn->weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->grad_weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}



int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("../mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_test.bin", y_test, TEST_SIZE);


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

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.grad_weights1);
    free(nn.grad_weights2);
    free(nn.grad_bias1);
    free(nn.grad_bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);
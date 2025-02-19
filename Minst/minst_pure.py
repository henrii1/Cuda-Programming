import numpy as np
from torchvision import datasets, transforms

# Load the data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

# Extract the data and labels
X_train = mnist_train.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_train = mnist_train.targets.numpy()
X_test = mnist_test.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_test = mnist_test.targets.numpy()

# print the shapes of the data
print("Train Data Shape:", X_train.shape)
print(y_train.shape)
print("Test Data Shape:", X_test.shape)
print(y_test.shape)

# Activation Functions
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# Linear layer
def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

def initialize_bias(output_size):
    return np.zeros((1, output_size))

def linear_forward(x, weight, bias):
    return x @ weight + bias

def linear_backward(dout, x, weights):
    grad_weights = x.T @ dout
    grad_bias = np.sum(dout, axis=0, keepdims=True)
    grad_input = dout @ weights.T
    return grad_input, grad_weights, grad_bias

# Softmax and cross-entropy loss
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    prob = softmax(y_pred)
    correct_logprobs = -np.log(prob[range(batch_size), y_true])
    loss = np.sum(correct_logprobs) / batch_size
    return loss


# define the model

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def __call__(self, x):
        batch_size = x.shape[0]
        fc1_input = x.reshape(batch_size, -1)
        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        return fc2_output, (fc1_input, fc1_output, relu_output)
    
    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache

        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output, relu_output, self.weights2)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x, self.weights1)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2
    
    def update(self, grad_weights1, grad_bias1, grad_weights2, grad_bias2, lr):
        self.weights1 -= lr * grad_weights1
        self.bias1 -= lr * grad_bias1
        self.weights2 -= lr * grad_weights2
        self.bias2 -= lr * grad_bias2



# Training the model

def train(model, X_train, y_train, X_test, y_test, batch_size, epochs, lr):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} out of {epochs}")
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            y_pred, cache = model(batch_X)
            loss = cross_entropy_loss(y_pred, batch_y)

            softmax_probs = softmax(y_pred)
            y_true = np.zeros_like(softmax_probs)
            y_true[range(batch_size), batch_y] = 1
            grad_output = (softmax_probs - y_true) / batch_size

            grad_weights1, grad_bias1, grad_weights2, grad_bias2 = model.backward(grad_output, cache)
            model.update(grad_weights1, grad_bias1, grad_weights2, grad_bias2, lr)

            if (i//batch_size) % 100 == 0:
                print(f"Iteration {i//batch_size}, Loss: {loss:.4f}")
        
        # Evaluate the model
        y_pred, _ = model(X_test)
        test_loss = cross_entropy_loss(y_pred, y_test)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    print("Training Complete")

if __name__ == "__main__":
    input_size = 784
    hidden_size = 256
    output_size = 10
    batch_size = 64
    epochs = 5
    lr = 0.01

    model = NeuralNetwork(input_size, hidden_size, output_size)
    train(model, X_train, y_train, X_test, y_test, batch_size, epochs, lr)

    

        
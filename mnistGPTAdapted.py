import numpy as np
from keras.datasets import mnist

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images and normalize pixel values
x_train_flat = x_train.reshape((x_train.shape[0], -1)) / 255.0
x_test_flat = x_test.reshape((x_test.shape[0], -1)) / 255.0

# One-hot encode labels
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]

# Manual split into training and validation sets
split_ratio = 0.9
split_index = int(len(x_train_flat) * split_ratio)

x_train_split, x_val_split = x_train_flat[:split_index], x_train_flat[split_index:]
y_train_split, y_val_split = y_train_one_hot[:split_index], y_train_one_hot[split_index:]

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Xavier initialization for weights
def xavier_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(1 / (input_size + output_size))

layer_sizes = [784,128,10]

# Initialize weights and biases
W,B = [],[]
for l in range(len(layer_sizes)-1):
    W.append(xavier_init(layer_sizes[l], layer_sizes[l+1]))
    B.append(np.zeros((1, layer_sizes[l+1])))

# Hyperparameters
learning_rate = 0.001
epochs = 50
batch_size = 64

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train_split), batch_size):
        # Mini-batch
        x_batch = x_train_split[i:i + batch_size]
        y_batch = y_train_split[i:i + batch_size]

        # Forward pass
        Z,A = [],[]
        Z.append(np.dot(x_batch, W[0]) + B[0])
        A.append(sigmoid(Z[0]))
        for l in range(1,len(layer_sizes)-2):
            Z.append(np.dot(A[l-1], W[l]) + B[l])
            A.append(sigmoid(Z[l]))
        Z.append(np.dot(A[-1], W[-1]) + B[-1])
        A.append(softmax(Z[-1]))

        # Backward pass
        delta_reversed = []
        delta_reversed.append(A[-1] - y_batch)
        for l in range(1,len(layer_sizes)-1):
            next_delta_layer = delta_reversed[l-1].dot(W[len(layer_sizes)-l-1].T) * (A[len(layer_sizes)-l-2] * (1 - A[len(layer_sizes)-l-2]))
            delta_reversed.append(next_delta_layer)
        delta = delta_reversed[::-1]

        dW,dB = [],[]
        dW.append(x_batch.T.dot(delta[0]))
        dB.append(np.sum(delta[0], axis=0, keepdims=True))
        for l in range(1,len(layer_sizes)-1):
            dW.append(A[l-1].T.dot(delta[l]))
            dB.append(np.sum(delta[l], axis=0, keepdims=True))

        # Update weights and biases
        for l in range(len(layer_sizes)-1):
            W[l] -= learning_rate * dW[l]
            B[l] -= learning_rate * dB[l]

    # Validation pass
    Z_val, A_val = [], []
    Z_val.append(np.dot(x_val_split, W[0]) + B[0])
    A_val.append(sigmoid(Z_val[0]))
    for l in range(1, len(layer_sizes) - 2):
        Z_val.append(np.dot(A_val[l - 1], W[l]) + B[l])
        A_val.append(sigmoid(Z_val[l]))
    Z_val.append(np.dot(A_val[-1], W[-1]) + B[-1])
    A_val.append(softmax(Z_val[-1]))

    predictions = np.argmax(A_val[-1], axis=1)
    true_labels = np.argmax(y_val_split, axis=1)

    accuracy = np.mean(predictions == true_labels)

    print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

# Test accuracy
Z_test, A_test = [], []
Z_test.append(np.dot(x_test_flat, W[0]) + B[0])
A_test.append(sigmoid(Z_test[0]))
for l in range(1, len(layer_sizes) - 2):
    Z_test.append(np.dot(A_test[l - 1], W[l]) + B[l])
    A_test.append(sigmoid(Z_test[l]))
Z_test.append(np.dot(A_test[-1], W[-1]) + B[-1])
A_test.append(softmax(Z_test[-1]))

test_predictions = np.argmax(A_test[-1], axis=1)
true_test_labels = np.argmax(y_test_one_hot, axis=1)
test_accuracy = np.mean(test_predictions == true_test_labels)

print(f"Test Accuracy: {test_accuracy:.4f}")
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

# Initialize weights and biases
input_size = x_train_flat.shape[1]
hidden_size = 128
output_size = 10

W1 = xavier_init(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = xavier_init(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hyperparameters
learning_rate = 0.01
epochs = 10
batch_size = 64

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train_split), batch_size):
        # Mini-batch
        x_batch = x_train_split[i:i + batch_size]
        y_batch = y_train_split[i:i + batch_size]

        # Forward pass
        z1 = np.dot(x_batch, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        # Backward pass
        delta2 = a2 - y_batch
        delta1 = delta2.dot(W2.T) * (a1 * (1 - a1))

        # print(delta2.shape,"delta2")
        # print(a1.shape,"a1")
        # print(a1.T.dot(delta2).shape,"dW2")

        dW2 = a1.T.dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        dW1 = x_batch.T.dot(delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Validation accuracy
    z1_val = np.dot(x_val_split, W1) + b1
    a1_val = sigmoid(z1_val)
    z2_val = np.dot(a1_val, W2) + b2
    a2_val = softmax(z2_val)

    predictions = np.argmax(a2_val, axis=1)
    true_labels = np.argmax(y_val_split, axis=1)
    accuracy = np.mean(predictions == true_labels)

    print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

# Test accuracy
z1_test = np.dot(x_test_flat, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = softmax(z2_test)

test_predictions = np.argmax(a2_test, axis=1)
true_test_labels = np.argmax(y_test_one_hot, axis=1)
test_accuracy = np.mean(test_predictions == true_test_labels)

print(f"Test Accuracy: {test_accuracy:.4f}")

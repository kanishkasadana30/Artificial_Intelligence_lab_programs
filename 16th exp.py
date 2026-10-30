

import numpy as np

# Input data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0], [1], [1], [1]])

# Initialize weights and biases
np.random.seed(1)
W1 = np.random.rand(2, 4)
B1 = np.zeros((1, 4))
W2 = np.random.rand(4, 1)
B2 = np.zeros((1, 1))

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Feed forward
hidden = sigmoid(np.dot(X, W1) + B1)
output = sigmoid(np.dot(hidden, W2) + B2)

# Display output
print("Predicted Output:")
print(np.round(output))



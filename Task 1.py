import numpy as np

np.random.seed(1)

W1 = np.random.randn(3, 4)
b1 = np.zeros(4)

W2 = np.random.randn(4, 1)
b2 = np.zeros(1)

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x):

    z1 = np.dot(x, W1) + b1

    hidden = relu(z1)

    z2 = np.dot(hidden, W2) + b2

    output = sigmoid(z2)

    return hidden, output


x = np.array([[0.5, -1.2, 3.0]])

hidden_values, output_probability = forward(x)

print("\n========== MATRIX SHAPES ==========")

print("Input shape:", x.shape)
print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)


print("\n========== WEIGHTS AND BIASES ==========")

print("W1:\n", W1)
print("b1:\n", b1)

print("W2:\n", W2)
print("b2:\n", b2)


print("\n========== FORWARD PASS ==========")

print("Input:", x)

print("Hidden layer values:", hidden_values)

print("Output probability:", output_probability)


if output_probability[0][0] >= 0.5:
    print("Prediction: 1")
else:
    print("Prediction: 0")

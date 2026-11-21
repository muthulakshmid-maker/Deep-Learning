import numpy as np
import matplotlib.pyplot as plt


def softmax(logits):
    # Subtract maximum for numerical stability
    exp_values = np.exp(logits - np.max(logits))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


logits = np.array([2.0, 1.0, 0.1])

softmax_output = softmax(logits)

print("\n========== SOFTMAX ==========")
print("Input:", logits)
print("Softmax output:", np.round(softmax_output, 3))
print("Sum:", np.sum(softmax_output))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


x = np.linspace(-6, 6, 200)

relu_values = relu(x)
sigmoid_values = sigmoid(x)
tanh_values = tanh(x)


plt.figure(figsize=(10, 6))

plt.plot(x, relu_values, label="ReLU")
plt.plot(x, sigmoid_values, label="Sigmoid")
plt.plot(x, tanh_values, label="Tanh")

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")

plt.legend()
plt.grid(True)

plt.savefig("activation_functions.png")
plt.show()


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.2, 0.7])

mse_value = mse(y_true, y_pred)

print("\n========== MEAN SQUARED ERROR ==========")
print("Actual values:", y_true)
print("Predicted values:", y_pred)
print("MSE:", mse_value)


def cross_entropy(y_true, y_prob):
    epsilon = 1e-15

    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)

    loss = -np.mean(
        np.sum(y_true * np.log(y_prob), axis=1)
    )

    return loss


y_true_class = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

y_prob = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
])

ce_value = cross_entropy(y_true_class, y_prob)

print("\n========== CROSS ENTROPY ==========")
print("Actual values:\n", y_true_class)
print("Predicted probabilities:\n", y_prob)
print("Cross Entropy:", ce_value)


confident_wrong = np.array([[0.01, 0.99]])

uncertain = np.array([[0.40, 0.60]])

actual = np.array([[1, 0]])

wrong_loss = cross_entropy(actual, confident_wrong)
uncertain_loss = cross_entropy(actual, uncertain)

print("\n========== LOSS COMPARISON ==========")

print("Confident wrong prediction loss:", wrong_loss)
print("Uncertain prediction loss:", uncertain_loss)


print("\n========== ACTIVATION AND LOSS SELECTION ==========")

print("1. House Price:")
print("   Output: Linear activation")
print("   Loss: MSE")

print("\n2. Spam Detection:")
print("   Output: Sigmoid activation")
print("   Loss: Binary Cross Entropy")

print("\n3. 5-Class Image Classification:")
print("   Output: Softmax activation")
print("   Loss: Categorical Cross Entropy")

import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset_raw = np.genfromtxt(r"C:\\Users\\adity\\Downloads\\heart.csv", dtype="str", delimiter=",")
print(dataset_raw.shape)

# Extract headers and data
headers = dataset_raw[0, :]
print(headers)
dataset = dataset_raw[1:, :].astype(float)
print(dataset)

# Extract features and labels
X = dataset[:, :13].T                   # Shape: (13, m)
Y = dataset[:, 13].reshape(1, -1)       # Shape: (1, m)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Split into training and testing data (80:20)
index = int(0.8 * X.shape[1])
X_train = X[:, :index]
X_test = X[:, index:]
Y_train = Y[:, :index]
Y_test = Y[:, index:]

# Confirm shapes
print("X_train shape", X_train.shape)
print("Y_train shape", Y_train.shape)
print("Number of training examples =", Y_train.shape[1])
print("-" * 40)
print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)
print("Number of testing examples =", Y_test.shape[1])

# Initialize weights and bias
def init_params(num_features):
    W = np.zeros((num_features, 1))
    b = 0
    return W, b

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation
def forward_prop(W, b, X):
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A

# Cost function
def calculate_loss(A, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m
    return np.squeeze(cost)

# Backward propagation
def backward_prop(A, X, Y):
    m = Y.shape[1]
    dW = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    return dW, db

# Parameter update
def update_params(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# Prediction
def predict(W, b, X):
    A = forward_prop(W, b, X)
    return (A >= 0.5).astype(float)

# Training function
def train(X, Y, num_iterations=10000, learning_rate=0.0001, print_cost=True):
    W, b = init_params(X.shape[0])
    costs = []

    for i in range(num_iterations):
        A = forward_prop(W, b, X)
        cost = calculate_loss(A, Y)
        dW, db = backward_prop(A, X, Y)
        W, b = update_params(W, b, dW, db, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")

    return W, b, costs

# Train the model
W, b, costs = train(X_train, Y_train)

# Plot the cost curve
plt.plot(costs)
plt.title("Cost over iterations")
plt.xlabel("Iterations (x100)")
plt.ylabel("Cost")
plt.show()

# Accuracy calculation
train_accuracy = 100 - np.mean(np.abs(predict(W, b, X_train) - Y_train)) * 100
test_accuracy = 100 - np.mean(np.abs(predict(W, b, X_test) - Y_test)) * 100

print(f"Train accuracy: {train_accuracy:.2f} %")
print(f"Test accuracy: {test_accuracy:.2f} %")

import numpy as np
import matplotlib.pyplot as plt
dataset_raw = np.genfromtxt(r"C:\\Users\\adity\Downloads\\heart.csv", dtype="str", delimiter=",")
print(dataset_raw.shape)
headers = dataset_raw[0, :]
print(headers)
dataset = dataset_raw[1:, :]
dataset = dataset.astype(float)
print(dataset)
X = dataset[:, :13]
Y = dataset[:, 13]
print(X.shape)
print(Y.shape)
X = X.T
print(X.shape)
print(Y.shape)
# get index to split data in 80:20 ratio
index = int(0.8 * X.shape[1])

# split the data
X_train = X[:, :index]
X_test = X[:, index:]

Y_train = Y[:index]
Y_test = Y[index:]
print("X_train shape", X_train.shape)
print("Y_train shape", Y_train.shape)
print("Number of training examples =", Y_train.shape[0])
print("-"*40)
print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)
print("Number of testing examples =", Y_test.shape[0])

def init_params(num_features):
    W = np.zeros((num_features, 1))
    b = 0
    return W, b

def sigmoid(x):
    s = 1/(1+np.exp(-1*x))
    return s

def forward_prop(W, b, X):
    Z = np.dot(W.T,X) + b
    A = sigmoid(Z)
    return A 

def calculate_loss(A, Y):
    m = Y.shape[0]
    cost = -1*np.sum((Y * np.log(A + 1e-8))+ (1-Y) * np.log(1-A + 1e-8))/m
    cost = np.squeeze(cost)

    return cost

def backward_prop(A, X, Y):
    m = Y.shape[0]
    dW = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m
    return dW, db

def update_params(W, b, dW, db, learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*db
    return W, b 

def predict(W, b, X):
  A = forward_prop(W, b, X)
  Y_pred = (A>=0.5)*1.0
  return Y_pred

def train(X, Y, num_iterations=10000, learning_rate=0.0001, print_cost=True):
    W, b = init_params(X_train.shape[0])
    costs = []

    for i in range(num_iterations):
        A = forward_prop(W,b,X)

        cost = calculate_loss(A,Y)

        dW, db = backward_prop(A,X,Y)

        W, b = update_params(W,b,dW,db,learning_rate)
        
        if i%100 == 0:
            costs.append(cost)

        if print_cost and i%100==0:
            print(f"Cost after {i+1} iteration : {cost}")

    return W, b, costs

W, b, costs = train(X_train,Y_train)

plt.plot(costs)
plt.show()

print("Train accuracy: {} %".format(100 - np.mean(np.abs(predict(W, b, X_train) - Y_train)) * 100))
print("Test accuracy: {} %".format(100 - np.mean(np.abs(predict(W, b, X_test) - Y_test)) * 100))




import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
X = X.T
y = y.reshape(-1, 1) 
y = y.T
X_train, X_test, y_train, y_test = train_test_split(X.T,y, test_size= 0.2, random_state= 42)        
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T
def initialize_param():
        np.random.seed(42)
    
        W1 = np.random.rand(15,30) * 0.01
        b1 = np.zeros((15,1))
        W2 = np.random.rand(10,15)* 0.01
        b2 = np.zeros((10,1))
        W3 = np.random.rand(1,10) * 0.01
        b3 = np.zeros((1,1))
  
        
        return W1, b1, W2, b2, W3, b3

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def relu_derivative(z):
    return z > 0

def Forward_prop(W1,b1,W2, b2, W3, b3,X):
  Z1 = np.dot(W1, X) + b1
  A1 = relu(Z1)
  Z2 = np.dot(W2, A1)+ b2
  A2 = relu(Z2)
  Z3 = np.dot(W3, A2) + b3
  A3 = sigmoid(Z3)
  return Z1,A1,Z2,A2,Z3,A3

def Backward_prop(A1,A2,A3, W2, W3, Z1, Z2, y, X):
    m = X.shape[1]
    
    dZ3 = A3 - y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
   
    return dW1, dW2, dW3, db1, db2, db3

def update_weights_bias( W1, b1, W2, b2, W3, b3,dW1,dW2,dW3,db1,db2,db3, learning_rate = 0.01):
   W1 = W1 - learning_rate * dW1
   b1 = b1 - learning_rate * db1
   W2 = W2 - learning_rate * dW2
   b2 = b2 - learning_rate * db2
   W3 = W3 - learning_rate * dW3
   b3 = b3 - learning_rate * db3
   
   return W1,b1,W2,b2,W3,b3

def compute_cost(A3, y):
    m = y.shape[1]
    cost = -1/m * np.sum(y * np.log(A3) + (1-y) * np.log(1 - A3))
    cost = np.squeeze(cost)  
    return cost

def Train_netwrok(X, y, number_iter = 1000, epoch = 64):
    W1, b1, W2, b2, W3, b3 = initialize_param()
    cost_history = []
    for i in range(epoch):
        for j in range(number_iter):
           Z1,A1,Z2,A2,Z3,A3 =  Forward_prop(W1,b1,W2, b2, W3, b3,X)
           cost = compute_cost(A3,y)
           dW1, dW2, dW3, db1, db2, db3 = Backward_prop(A1,A2,A3, W2, W3, Z1, Z2, y, X)
           W1,b1,W2,b2,W3,b3 = update_weights_bias( W1, b1, W2, b2, W3, b3,dW1,dW2,dW3,db1,db2,db3, learning_rate = 0.01)
           
           if j % 100 == 0:
               print("Epoch:", i, "Iteration:", j, "Cost:", cost)
               cost_history.append(cost)
    return W1,b1,W2,b2,W3,b3, cost_history
  
W1,b1,W2,b2,W3,b3,cost_history = Train_netwrok(X_train, y_train)    

def predict(W1, b1, W2, b2, W3, b3, X):
    _, _, _, _, _, A3 = Forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = (A3 > 0.5).astype(int)
    return predictions

predictions = predict(W1, b1, W2, b2, W3, b3, X_test)

def get_accuracy(predictions, y_test):
    return np.sum(predictions == y_test) / y_test.shape[1]

get_accuracy(predictions, y_test)

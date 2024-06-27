# Breast Cancer Prediction Using Neural Network
This project implements a neural network to predict breast cancer using the breast cancer dataset from sklearn. The neural network is built using numpy and includes forward propagation, backward propagation, and parameter updates to train the model.

# Dataset
The dataset used is the breast cancer dataset from sklearn. It consists of 30 features and a target variable indicating the presence or absence of breast cancer.

# Prerequisites
- numpy
- sklearn

# Code Structure
1. Data Preprocessing
2. Neural Network Initialization
3. Forward Propagation
4. Backward Propagation
5. Parameter Update
6. Cost Calculation
7. Training the Network : the neural network is trained for a specified number of epochs and iterations
      #      def Train_network(X, y, number_iter=1000, epoch=64):
                W1, b1, W2, b2, W3, b3 = initialize_param()
                cost_history = []
            
                for i in range(epoch):
                    for j in range(number_iter):
                        Z1, A1, Z2, A2, Z3, A3 = Forward_prop(W1, b1, W2, b2, W3, b3, X)
                        cost = compute_cost(A3, y)
                        dW1, dW2, dW3, db1, db2, db3 = Backward_prop(A1, A2, A3, W2, W3, Z1, Z2, y, X)
                        W1, b1, W2, b2, W3, b3 = update_weights_bias(W1, b1, W2, b2, W3, b3, dW1, dW2, dW3, db1, db2, db3)
                        
                        if j % 100 == 0:
                            print("Epoch:", i, "Iteration:", j, "Cost:", cost)
                            cost_history.append(cost)
            
                return W1, b1, W2, b2, W3, b3, cost_history
8. Prediction:
   def predict(W1, b1, W2, b2, W3, b3, X):
    _, _, _, _, _, A3 = Forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = (A3 > 0.5).astype(int)
    return predictions
9. Accuracy:

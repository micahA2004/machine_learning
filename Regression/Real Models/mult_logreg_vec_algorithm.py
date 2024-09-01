"""
Changes to make:
    Handling categorical features using one-hot encoding or label encoding.
    Optimizing speed:
        Improve data handling with pandas/numpy.
        Experiment with advanced optimization algorithms like Adam.
    Usability/customizability:
        Allow the script to handle various logistic regression tasks with minimal modifications.
        Implement features to save predictions to a new column in the DataFrame for evaluation.
    Readability:
        Refactor code to enhance clarity and maintainability.

    This program is designed for logistic regression using gradient descent with vectorization.
    It should be adaptable to different datasets with minimal adjustments, primarily related
    to how data is imported and prepared.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time as t
from sklearn.metrics import accuracy_score

start = t.time()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Model: logistic regression prediction
model = lambda w, X, b: sigmoid(np.dot(X, w) + b)

# Cost function for logistic regression
def cost(w, X, b, Y):
    m = len(Y)
    predicted = model(w, X, b)
    cost = -(1/m) * np.sum(Y * np.log(predicted) + (1 - Y) * np.log(1 - predicted))
    return cost

# Gradient calculation
def gradient(w, X, b, Y):
    m = len(Y)
    predicted = model(w, X, b)
    error = predicted - Y
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

# Gradient descent
def gradient_descent(w, X, b, Y, learning_rate, iteration_num, interval):
    x_iteration_num = np.arange((iteration_num + 1) // interval)
    y_cost_list = np.zeros((iteration_num + 1) // interval)
    
    li = 0
    
    for i in range(iteration_num):
        dj_dw, dj_db = gradient(w, X, b, Y)
        
        w_temp = w - learning_rate * dj_dw
        b_temp = b - learning_rate * dj_db
        
        w, b = w_temp, b_temp
      
        if i % interval == 0 or i == 0:
            cost_iter = cost(w, X, b, Y)
            y_cost_list[li] = cost_iter
            li += 1

    plt.plot(x_iteration_num, y_cost_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost of Logistic Model vs. Number of Iterations")
    plt.show()
    
    final_cost = y_cost_list[-1]
    
    return w, b, final_cost

# Mean normalization
def mean_normalization(w, X):
    shape = X.shape
    if len(w) != shape[0]:
        X = X.T
      
    normalized = []
    for (i, vec) in enumerate(X):
        mean = np.mean(vec)
        rang = max(vec) - min(vec)  
        vec = (vec - mean) / rang
        normalized.append(vec)
        
    return np.array(normalized)

### HANDLING DATA ###
# Simulated dataset (replace with actual file path)
df = pd.read_csv("C:\\Users\\Micah\\Desktop\\machine_learning\\Datasets\\Simulated_Logistic_Dataset.csv")

# Encoding categorical feature (if any)
df['Category'] = df['Category'].map({'Yes': 1, 'No': 0})

test_ratio = 0.20
train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)

input_data_train = train_df.drop(["Outcome"], axis=1)
target_data_train = train_df["Outcome"]
input_data_test = test_df.drop(["Outcome"], axis=1)
target_data_test = test_df["Outcome"]

X = input_data_train.to_numpy()
Y = target_data_train.to_numpy()
X_test = input_data_test.to_numpy()
Y_test = target_data_test.to_numpy()

# Initializing weights and bias
n = X.shape[1]
w = np.zeros(n)
b = 0

# Normalizing features
norm_X_train = mean_normalization(w, X)
norm_X_test = mean_normalization(w, X_test)
X = norm_X_train.T
X_test = norm_X_test.T

# Training parameters
alpha = 0.01  # Learning rate
iterations = 10000
interval = 100

w, b, final_cost = gradient_descent(w, X, b, Y, alpha, iterations, interval)

# Prediction and evaluation
def predict(X, w, b):
    probabilities = model(w, X, b)
    return [1 if p >= 0.5 else 0 for p in probabilities]

Y_pred_train = predict(X, w, b)
Y_pred_test = predict(X_test, w, b)

train_accuracy = accuracy_score(Y, Y_pred_train) * 100
test_accuracy = accuracy_score(Y_test, Y_pred_test) * 100

print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Testing accuracy: {test_accuracy:.2f}%")

end = t.time()
t_sec = end - start
print(f"Time this script took: {t_sec:.2f} seconds.")


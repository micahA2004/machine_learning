"""
Changes to make:
    Dealing with features with discrete labels, or more efficient normalization
    Speed
        Data handling using pandas/numpy
        Actual algorithm
        Thinking of implementing non-negative matrix factorization to help with dataframe parsing
    Usability/customizability
        I want this script to be able to handle custom situations
        Could write the predicted values to a new column in dataframe to assist evaluation in real testing cases
    Readability
        Overall cleaning up everything and making program more professional
    
    This program should work well with a multitude of datasets using gradient descent
    for multiple linear regression w/ vectorization. Most significant modification a user
    should need to make is how they pull in the training and testing data for model.
"""

##deal with feature normalization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as t
import statistics as stat
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

start = t.time()

model = lambda w,X,b: np.dot(X,w) + b #calculates model with given parameters and matrix X

#calculates the standard cost for linear regression gradient descent
def cost(w,X,b,Y):
    
    m = len(Y)
    avg = 1 / (m * 2)
    
    predicted = model(w, X, b)
    inner = predicted - Y
    #sums together the matrix produced, provides more comprehensive cost
    sse = np.sum(inner ** 2) 
    cost = sse * avg

    return cost

#gets the partials of each parameter involved in the model
def gradient(w, X, b, Y):
    m = len(Y)
    avg = 1 / m #avg is only 1/m now b/c we took deriv (chain rule)
    
    predicted = model(w, X, b)
    inner = predicted - Y
    
    #just the partial derivative of cost in respect to b
    dj_db = avg * np.sum(inner) #returns a scalar
    #partial derivative of cost in respect to w
    dj_dw = avg * np.dot(X.T,inner) #returns a vector of length n, sae as w
    
    
    return dj_dw, dj_db

#make sure the iteration num is a multiple of 100, need to make this more robust later

def gradient_descent(w, X, b, Y, learning_rate, iteration_num, interval):
    """
    Given a learning rate, number of iterations to run through, and an interval to 
    calculate cost at, this will return a vector of w values, a scalar b value, and then
    the final cost calculated to demonstrate the effectivity of the model found by
    gradient descent. Also returns a plot of the number of iterations vs. cost.
    """
    
    #this is purely for graphing purposes, adding 1 to length because I want 
    #the initial cost to also be in the graph
    x_iteration_num = np.arange((iteration_num + 1) // interval)
    y_cost_list = np.zeros((iteration_num + 1) // interval)
    
    li = 0 #counter for the interval
    
    for i in range(iteration_num):
        dj_dw, dj_db = gradient(w, X, b, Y)
        
        w_temp = w - learning_rate * dj_dw
        b_temp = b - learning_rate * dj_db
        
        #simultaneously assigning new values to maintain accuracy
        w, b = w_temp, b_temp
      
        if i % interval == 0 or i == 0:
            cost_iter = cost(w,X,b,Y)
            y_cost_list[li] = cost_iter
            li += 1
        

    plt.plot(x_iteration_num,y_cost_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Cost of Predicted Model vs. Number of Iterations")
    plt.show()
    
    c = y_cost_list[-1]
    
    return w, b, c

def mean_normalization(w, X):
    shape = X.shape
    if len(w) != shape[0]:
        X = X.T
      
    new = []
    for (i, vec) in enumerate(X):
        mean = stat.mean(vec)
        rang = max(vec) - min(vec)  
        vec = (vec - mean) / (rang)
        new.append(vec)
        
    new = np.array(new)
    return new
    

###TAKING CARE OF DATA PORTION###
df = pd.read_csv("C:\\Users\\Micah Andrejczak\\Desktop\machine_learning\\Datasets\\Student_Performance.csv")

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

test_ratio = 0.20
train_df, test_df = train_test_split(df, test_size= test_ratio, random_state=42)  # 60% training, 40% testing

# tdf,cdf = train_test_split()
input_data_train = train_df.drop(["Performance Index"], axis = 1)
target_data_train = train_df["Performance Index"]
input_data_test = test_df.drop(["Performance Index"], axis = 1)
target_data_test = test_df["Performance Index"]

X = input_data_train.to_numpy()
Y = target_data_train.to_numpy()
X_test = input_data_test.to_numpy()
Y_test = target_data_test.to_numpy()

#getting dimensions of data matrix
shape = X.shape

n = shape[1]

#initial guesses for parameters
w = np.zeros(n)

b = 0

norm_X_train = mean_normalization(w, X)
norm_X_test = mean_normalization(w, X_test)
X_test = norm_X_test.T
X = norm_X_train.T
#alpha was found to be the fastest learning rate that doesn't allow divergence through testing
alpha = 0.1
#modify either of these, higher iterations results in lower cost
iterations = 10000
interval = 100

w,b,c = gradient_descent(w, X, b, Y, alpha, iterations, interval)

#Quick function to make it easier to print out final model with proper coefficients
def create_model_string(w, b):
    terms = [f"{w[i]:.6f}(x{i+1})" for i in range(len(w))]
    terms_str = " + ".join(terms)
    return f"{terms_str} + {b:.6f}"


#Prints the model rep
model_str = create_model_string(w, b)
print(f"Final model: y = {model_str}\n")
print(f"Final cost is : {c}.\n")


end = t.time()

t_sec = end - start

#Trying to plot all of the feature graphs
# feature_names = input_data.columns
# for i in range(n):
#     plt.figure()
#     plt.scatter(X[:, i], Y, label='Actual Data', color='blue')
#     plt.plot(X[:, i], model(w, X, b), label='Model Prediction', color='red')
#     plt.xlabel(f"Feature {feature_names[i]}")
#     plt.ylabel("Performance Index")
#     plt.title(f"Feature {feature_names[i]} vs. Performance Index")
#     plt.legend()
#     plt.show()
    
print(f"Time this script takes in seconds:\n {t_sec} seconds.\n")

# Writing the execution time to a file


#at this point we have our regression model with normalized data, and we have our X_test matrix and Y_test vector
#Let's now use the model to calculate the predicted y value and calculate percentages to evaluate accuracy

#we will just make this with scalars first
#remember percent error should just be (found value - actual value) / (actual value)
def percent_error(exact, approximate):
    num = approximate - exact
    err = abs(num) / (exact)
    return err

Y_predicted = model(w, X_test, b)
Y_test = Y_test

err = percent_error(Y_test,Y_predicted)
overall_err = np.sum(err) / len(err)


percent_err = overall_err * 100

testing_string = f"Final percent error of model vs. testing data is {percent_err:.3f}%.\n\n"
conclusion = f"This means that the percent accuracy of this model is {(100-percent_err):.3f}%. "

print(testing_string + conclusion)

#writes to a .txt file
with open("execution_time_w_num_Iterations.txt", "a") as file:
    file.write(f"{t_sec}, {iterations}, {test_ratio}, {percent_err}\n")
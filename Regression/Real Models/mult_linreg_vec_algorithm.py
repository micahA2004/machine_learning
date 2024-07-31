import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as t

start = t.time()


df = pd.read_csv("C:\\Users\\Micah Andrejczak\\Desktop\machine_learning\\Datasets\\Student_Performance.csv")

#dropping this from dataframe because I am not sure how to deal with yes or no at this moment, or how to weight it
#probably could just convert these vals to 1 or 0
df = df.drop(["Extracurricular Activities"], axis = 1)

#input data is everything except for target
input_data = df.drop(["Performance Index"], axis = 1)
target_data = df["Performance Index"]

X = input_data.to_numpy()
Y = target_data.to_numpy()

#getting dimensions of data matrix
shape = X.shape

n = shape[1]

#initial guesses for parameters
w = np.zeros(n)
b = 0

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

#alpha was found to be the fastest learning rate that doesn't allow divergence through testing
alpha = 0.00025
#modify either of these, higher iterations results in lower cost
iterations = 100000
interval = 1000

w,b,c = gradient_descent(w, X, b, Y, alpha, iterations, interval)

#Quick function to make it easier to print out final model with proper coefficients
def create_model_string(w, b):
    
    terms = [f"{w[i]}:.6f(x{i+1})" for i in range(len(w))]
    terms_str = " + ".join(terms)
    return f"{terms_str} + {b}"

#Prints the model rep
model_str = create_model_string(w, b)
print(f"Final model: y = {model_str}")
print(f"Final cost is : {c}.")


end = t.time()

t_sec = end - start

#Trying to plot all of the feature graphs
feature_names = input_data.columns
for i in range(n):
    plt.figure()
    plt.scatter(X[:, i], Y, label='Actual Data', color='blue')
    plt.plot(X[:, i], model(w, X, b), label='Model Prediction', color='red')
    plt.xlabel(f"Feature {feature_names[i]}")
    plt.ylabel("Performance Index")
    plt.title(f"Feature {feature_names[i]} vs. Performance Index")
    plt.legend()
    plt.show()
    
print(f"Time this script takes in seconds:\n {t_sec} seconds.")
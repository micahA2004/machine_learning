import numpy as np 
import pandas as pd
import multiprocessing as mp

# basic model for linear regression
model = lambda x,w,b : (w*x) + b

#function to calculate cost of model
def cost(x,y,w,b):
    if len(x) != len(y):
        raise Exception("Arrays must be same length")
    else:
        l = len(x)
        error = 0
        for i in range(l):
            predicted = model(x[i],w,b)
            actual = y[i]
            err = (predicted - actual) ** 2
            error += err
        avg = (1 / (l * 2))
        cost = error * avg
        return cost

def grad_descent(x,y,w,b,alpha):
    tol = 1e-6
    m = len(x)
    diff_w = diff_b = 1
    c = 0
    
    while True:
        s_deriv_w = s_deriv_b = 0
        
        for i in range(m):
            predicted = (model(x[i],w,b))
            actual = y[i]
            err_b = (predicted - actual)
            err_w = err_b * x[i]
            s_deriv_w += err_w
            s_deriv_b += err_b
        
        avg = (1 / m)
        w_deriv = s_deriv_w * avg
        b_deriv = s_deriv_b * avg
            
        w_temp = w - (alpha * w_deriv)
        b_temp = b - (alpha * b_deriv)
            
        diff_w = abs(w - w_temp)
        diff_b = abs(b - b_temp)
        
        c += 1
        # if c == 20:
        #     break
        print(f"Iteration {c}, w = {w} and b = {b}.")
        
        if diff_w > tol:
            w = w_temp
        if diff_b > tol:
            b = b_temp
            
        if (diff_w < tol) and (diff_b < tol):
            break
    
    print(diff_w)
    print(diff_b)
    return w,b,c

df = pd.read_csv("C:\\Users\\Micah Andrejczak\\Downloads\\archive\\train.csv")
input_data = df["x"]
target_data = df["y"]
x = input_data.to_numpy()
y = target_data.to_numpy()

#guesses
w = 1
b = -2
alpha= 0.0001 #alpha value for gradient descent
w,b,c = grad_descent(x,y,w,b,alpha)
w_sci = 1.000778257535692
b_sci = -0.12015653187363284
cost_sci = cost(x,y,w_sci,b_sci)
cost = cost(x,y,w,b)

print(f"w = {w}, b = {b} for optimal model.")
print(f"w = {w_sci}, b = {b_sci} for scikit model.")
print(f"The cost difference between my model and scikitlearn's regression model is {abs(cost - cost_sci)}.")
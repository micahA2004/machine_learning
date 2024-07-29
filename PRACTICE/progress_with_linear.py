import numpy as np

"""
Let's figure this out.

Features for analysis
   x1              x2                  x3                 x4             y(TARGET)
   2               3                   4                  5              17                2 items
   7               1                   9                  2              22             in each feature
   
W values (1 w for each feature involved in linear regression)
   w1              w2                  w3                 w4      
   1               2                   3                  4                  

Let b = 4

y_hat = w1x1 + w2x2 + w3x3 + w4x4 + b
      = (x dot w)  + b                                                              

If we followed the first model:
    w1 = 1
    x1 = [2,7]
    w1x1 = 2 + 7 = 9
    
    w2 = 2
    x2 = [3,1]
    w2x2 = 2 * 3 + 2 * 1 = 6 + 2 = 8
    
    w3 = 3
    x3 = [4,9]
    w3x3 = 3 * 4 + 3 * 9 = 12 + 27 = 39
    
    w4 = 4
    x4 = [5,2]
    w4x4 = 4 * 5 + 4 * 2 = 20 + 8 = 28
    
    w1x1 + w2x2 + w3x3 + w4x4 = 9 + 8 + 39 + 28 = 84
    = 84
    
    wdotx + b = 88
    
If we followed the second model:
    yhat = X . w + b
    
 X = [[2 7]
     [3 1]
     [4 9]
     [5 2]]
    
 w = [1 2 3 4] or [1
                   2
                   3
                   4]
                     
    

"""



# Features for analysis: 2 sets of data with 4 features each
# Feature matrix X (transposed for proper dot product)
x1 = [2, 7]
x2 = [3, 1]
x3 = [4, 9]
x4 = [5, 2]
y = [17, 22]

# Combine features into a single matrix
X = np.array([x1, x2, x3, x4]).T  # Transposed to get the shape (2, 4)
print("Feature matrix X:")
print(X)
print("\n")

# Weights for each feature
w1 = 1
w2 = 2
w3 = 3
w4 = 4
w = np.array([w1, w2, w3, w4])
print("Weight vector w:")
print(w)
print("\n")

# Bias term
b = 4

# Calculate the model output (predicted values) using dot product
d = np.dot(X, w) + b
print("Predicted values (y_hat):")
print(d)
print("\n")

# Actual target values
Y = np.array([17, 22])
print("Actual target values Y:")
print(Y)

#START OF THE ACTUAL CODE
model = lambda w,X,b: np.dot(X,w) + b

def cost(w,X,b,Y):
    #cost is just 1/2m * (sum of (predicted - actual)**2)
    shape = Y.shape
    m = shape[0]
    
    predicted = model(w,X,b)
    sse = np.sum((predicted - Y) ** 2) # sse applied to entire matrix
    avg = 1 / (m * 2)
    cost = sse * avg
    # for (i, xi) in enumerate(X):
    #     predicted = model(w, xi, b)
    #     actual = Y[i]
    #     sse = (predicted - actual) ** 2
    #     error += sse
        
    # avg = (1) / (m * 2)
    # error *= avg
    return cost

c = cost(w,X,b,Y)
print("\nInitial cost:")
print(c)

#partial in respect to w vector should be (predicted vector - actual vector) * x vector
def partials(w, X, b, Y):
    
    shape = Y.shape
    m = shape[0]
    
    avg_deriv = (1/m)
    predicted = model(X,w,b)
    Y_diff = predicted - Y
    dJ_dw = avg_deriv * np.dot(X, Y_diff )
    


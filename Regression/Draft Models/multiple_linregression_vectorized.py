import numpy as np

"""
Let's figure this out.

n -> feature number (column), j refers to specific iteration of feature
m -> data set number (row), i refers to specific iteration of data
Features for analysis
   x1              x2                  x3             
   2               11                  4             
   7               13                  7             
   9               12                  5             
   3               9                   2             
   1               8                   4             

In this case, n ranges from 1-3 (0-2 in Python)
              m ranges from 1-5 (0-4 in Python)
              
y(TARGET)
   22
   23
   30
   19
   21

W values (1 w for each feature involved in linear regression)
   w1              w2                  w3      
   3               7                   1                  

Let b = 1

y_hat = w1x1 + w2x2 + w3x3 + b
      = (x dot w)  + b                                                              

If we followed the nonvectorized approach:
    w1 = 3
    x1 = [2,7,9,3,1]
    w1x1 = 3 * 2 + 3 * 7 + 3 * 9 + 3 * 3 + 3 * 1 = 6 + 21 + 27 + 9 + 3 = 66
    
    w2 = 7
    x2 = [11,13,12,9,8]
    w2x2 = 7 * 11 + 7 * 13 + 7 * 12 + 7 * 9 + 7 * 8 = 77 + 91 + 84 + 63 + 56 = 371
    
    w3 = 1
    x3 = [4,7,5,2,4]
    w3x3 = 1 * 4 + 1 * 7 + 1 * 5 + 1 * 2 + 1 * 4 = 4 + 7 + 5 + 2 + 4 = 22
    
    w1x1 + w2x2 + w3x3 = 66 + 371 + 22 = 459
    = 459
    
    y_hat = 459 + (num_data_points * b) = 459 + (5 * 1)
        = 464
    
If we followed the vectorized approach:
    yhat = X . w + b
    
 X = [[ 2 11  4]
      [ 7 13  7]
      [ 9 12  5]
      [ 3  9  2]
      [ 1  8  4]]
    
 w = [3 7 1] or [3
                 7
                 1]
 
 X . w = [87, 119, 116, 74, 63]
 y^ = (X . w) + 1 = [88, 120, 117, 75, 64]
 sum(y^) = 464
                     
"""

x1 = [2, 7, 9, 3, 1]
x2 = [11, 13, 12, 9, 8]
x3 = [4, 7, 5, 2, 4]

y = [22, 23, 30, 19, 21]

# Combine features into a single matrix
X = np.array([x1, x2, x3]).T  # Transposed to get the shape (5, 3)
print("Feature matrix X:")
print(X)
print("\n")

# Weights for each feature
w1 = 3
w2 = 7
w3 = 1

w = np.array([w1, w2, w3]).T
print("Weight vector w:")
print(w)
print("\n")

# Bias term
b = 1

# Calculate the model output (predicted values) using dot product
d = np.dot(X, w) + b
print("Predicted values (y_hat):")
print(d)
suma = np.sum(d)
print(f"Summed model value is {suma}.")
print("\n")

#nonvectorized model to check if we are doing things correctly
y_hat_nv = []
for i in range(len(x1)):
    y_hat_v = w1 * x1[i] + w2 * x2[i] + w3 * x3[i] + b
    y_hat_nv.append(y_hat_v)
    
print("Predicted values (y_hat) using non-vectorized approach:")
print(y_hat_nv)
print(f"Summed model value is {np.sum(y_hat_nv)}.")
print("\n")

# Actual target values
Y = np.array(y)
print("Actual target values Y:")
print(Y)
print("\n")

#START OF ACTUAL HIGH LEVEL PROCESS, before was verification
model = lambda w,X,b: np.dot(X,w) + b

def cost(w,X,b,Y):
    #cost is just 1/2m * (sum of (predicted - actual)**2)
    shape = Y.shape
    m = max(shape)
    
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

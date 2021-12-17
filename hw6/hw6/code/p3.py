# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-11 16:49:22
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-11 22:49:54
import numpy as np
import gd
import utils
import matplotlib.pyplot as plt


# X: m x 2 matrix, X[i,:] is the 2D feature vector of the i-th sample
X = np.array([[1,1.5], 
			  [1.2, 2.5], 
			  [1,3.5], 
			  [2,2.25], 
			  [1.8, 3], 
			  [2.5,4], 
			  [3,1.9], 
	          [1.5, .5], 
	          [2.5, .8], 
	          [2.8, .3], 
	          [3.2, .3], 
	          [3, .8], 
	          [3.8, 1], 
	          [4,2], 
	          [1.8,1.8]])
# y: m-D vector, y[i] is the label of the i-th sample
y = np.append(np.ones((7,)), -np.ones((8,)))

# append a constant 1 to each feature vector, so X is now a m x 3 matrix
X = np.append(X, np.ones((15,1)), axis=1)
print(X[0])
# Xy[i,:] = X[i,:] * y[i]
Xy = X * y.reshape((-1,1))

# X.shape = (15,3), Xy.shape = (15, 3), y.shape = (15,)
print (X.shape, Xy.shape, y.shape)

# sigmoid function
def sigmoid(z):
	#   START OF YOUR CODE
	return 1 / (1 + np.exp(-z))
	#   END OF YOUR CODE
def fp(w):
	#   START OF YOUR CODE
	res = np.array([0,0,0])
	for i in range(15):
		k = 1-sigmoid(np.matmul(np.multiply(y[i], X[i]), w))
		res = res - np.multiply(y[i], X[i]) * k
	return res
	#   END OF YOUR CODE

# minimize f by gradient descent
#   START OF YOUR CODE
stepsize = 0.01
w0 = np.array([1.,1.,1.])
count = 0
w_traces = [w0]
w = w0
while count < 100000 and np.linalg.norm(fp(w)) > 1e-5:
	count += 1
	w = w - fp(w) * stepsize
	w_traces.append(w)
#   END OF YOUR CODE

# compute the accuracy on the training set
w = w_traces[-1]
y_hat = 2*(X@w > 0)-1

accuracy = sum(y_hat == y) / float(len(y))
print(f"accuracy = {accuracy}")

# visualization
plt.figure(figsize=(4, 4))

plt.scatter(*zip(*X[y>0, 0:2]), c='r', marker='x')
plt.scatter(*zip(*X[y<0, 0:2]), c='g', marker='o')

# plot the decision boundary w[0] * x1 + w[1] * x2 + w[0] = 0
x1 = np.array([min(X[:, 0]), max(X[:, 0])])
x2 = -(w[0] * x1 + w[2]) / w[1]
plt.plot(x1, x2, 'b-')

plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.savefig(r'C:\Users\Lenovo\Desktop\hw6\figures\p3.pdf')

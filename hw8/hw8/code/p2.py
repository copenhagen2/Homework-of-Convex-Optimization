# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-22 16:29:28
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-25 00:38:53
import numpy as np
import newton
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

# Xy[i,:] = X[i,:] * y[i]
Xy = X * y.reshape((-1,1))

# X.shape = (15,3), Xy.shape = (15, 3), y.shape = (15,)
print (X.shape, Xy.shape, y.shape)

# sigmoid function
def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def sigmoid_p(z):
	s = sigmoid(z)
	return s * (1 - s)

def f(w):
	return -np.sum(np.log(sigmoid(Xy@w)))

def fp(w):
	return -(1-sigmoid(Xy@w)) @ Xy

def fpp(w):
	#   START OF YOUR CODE
	A = sigmoid_p(Xy@w)
	B = np.array([A[i]*X[i] for i in range(15)])
	return X.T @ B
	#   END OF YOUR CODE

# minimize f by damped Newton
w0 = np.array([1.0,1.0,0.0])
path = 'C:\\Users\\Lenovo\\Desktop\\hw8\\figures\\2\\'

w_traces, stepsize_traces, num_iter_inner = newton.damped_newton(f, fp, fpp, w0, alpha=0.1, beta=0.7)
ws = w_traces[-1]
fs = f(ws)

print()
print("Damped Newton's method")
print('  number of iterations in outer loop:', len(w_traces)-1)
print('  total number of iterations in inner loop:', num_iter_inner)
print('  solution:', ws)
print('  value:', fs)


def gap(w):
	return f(w) - fs

utils.plot(gap, w_traces, path+'dnt_gap.pdf')

fig = plt.figure(figsize=(3.5,2.5))
plt.plot(stepsize_traces, '-o', color='blue')
plt.xlabel('iteration (k)')
plt.ylabel('stepsize')
plt.tight_layout(pad=0.1)
fig.savefig(path+'dnt_ss.pdf')

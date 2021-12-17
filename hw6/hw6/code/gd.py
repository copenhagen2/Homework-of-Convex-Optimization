# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-11 16:49:22
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-11 22:02:15
import numpy as np

def gd_const_ss(fp, x0, stepsize, tol=1e-5, maxiter=100000):
	"""
	fp: function that takes an input x and returns the derivative of f at x
	x0: initial point in gradient descent
	stepsize: constant step size used in gradient descent
	tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
	     when the 2-norm of the gradient is smaller than tol
	maxiter: maximum number of iterations in gradient descent.

	This function should return a list of the sequence of approximate solutions
	x_k produced by each iteration
	"""
	x_traces = [x0]
	x = x0
	count = 0
	#   START OF YOUR CODE
	while count < maxiter and np.linalg.norm(fp(x)) > tol:
		count += 1
		x = x - fp(x)*stepsize
		x_traces.append(x) 
	#	END OF YOUR CODE
	
	return x_traces 

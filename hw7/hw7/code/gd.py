# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-20 08:55:20
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-20 20:26:19
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
	x_traces = [np.array(x0)]
	x = np.array(x0)
	count = 0
	#   START OF YOUR CODE
	while count < maxiter and np.linalg.norm(fp(x)) > tol:
		count += 1
		x = x - fp(x) * stepsize
		x_traces.append(x) 
	#	END OF YOUR CODE

	return x_traces 

def gd_armijo(f, fp, x0, initial_stepsize=1.0, alpha=0.5, beta=0.5, tol=1e-5, maxiter=100000):
	"""
	f: function that takes an input x an returns the value of f at x
	fp: function that takes an input x and returns the derivative of f at x
	x0: initial point in gradient descent
	initial_stepsize: initial stepsize used in backtracking line search
	alpha: parameter in Armijo's rule 
				f(x - t * f'(x)) > f(x) - t * alpha * ||f'(x)||^2
	beta: constant factor used in stepsize reduction
	tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
	     when the 2-norm of the gradient is smaller than tol
	maxiter: maximum number of iterations in gradient descent.

	This function should return a list of the sequence of approximate solutions
	x_k produced by each iteration and the total number of iterations in the inner loop
	"""
	x_traces = [np.array(x0)]
	stepsize_traces = []
	tot_num_inner_iter = 0

	x = np.array(x0)
	#   START OF YOUR CODE
	count = 0
	while count < maxiter and np.linalg.norm(fp(x)) > tol:
		count += 1
		t = initial_stepsize
		while f(x)-f(x-t*fp(x)) < alpha* t * fp(x)@fp(x).T:
			tot_num_inner_iter += 1
			t *= beta
		stepsize_traces.append(t)
		x -= t * fp(x)
		x_traces.append(x)
		print(x)
	#	END OF YOUR CODE

	return x_traces, stepsize_traces, tot_num_inner_iter
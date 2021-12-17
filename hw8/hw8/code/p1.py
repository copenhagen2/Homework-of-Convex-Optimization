# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-11-22 16:29:28
# @Last Modified by:   Your name
# @Last Modified time: 2021-11-24 17:00:42
import numpy as np
import newton
import utils 
import matplotlib.pyplot as plt


def f(x):
	return f_2d(x[0], x[1])

def fp(x):
	#   START OF YOUR CODE
	return np.array([np.exp(x[0]+3*x[1]-0.1)+np.exp(x[0]-3*x[1]-0.1)-np.exp(-x[0]-0.1), 3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1)])
	#	END OF YOUR CODE

def fpp(x):
	#   START OF YOUR CODE
	x1 = x[0]
	x2 = x[1]
	g2 = [[np.exp(x1+3*x2-0.1)+np.exp(x2-3*x2-0.1)+np.exp(-x1-0.1), 3*np.exp(x1+3*x2-0.1)-3*np.exp(x2-3*x2-0.1)],
		[3*np.exp(x1+3*x2-0.1)-3*np.exp(x1 - 3*x2 - 0.1), 9*np.exp(x1+3*x2-0.1)+9*np.exp(x1 - 3*x2 - 0.1)]]
	return np.array(g2)
	#	END OF YOUR CODE

def f_2d(x1, x2):
	return np.exp(x1+3*x2-0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1-0.1)

# use the value you find in HW7
f_opt = 2 * np.exp(0.5 * np.log(2)-0.1)

def gap(x):
	return f(x) - f_opt

x0 = np.array([1.5,1.0])
path = 'C:\\Users\\Lenovo\\Desktop\\hw8\\figures\\1\\'

#### Newton
x_traces = newton.newton(fp, fpp, x0)
f_value= f(x_traces[-1])


print()
print("Newton's method")
print('  number of iterations:', len(x_traces)-1)
print('  solution:', x_traces[-1])
print('  value:', f_value)

utils.plot_traces_2d(f_2d, x_traces, path+'nt_traces.pdf')
utils.plot(gap, x_traces, path+'nt_gap.pdf')
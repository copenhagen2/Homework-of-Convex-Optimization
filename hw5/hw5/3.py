import cvxpy as cp
import numpy as np

m = 3
n = 2
A = np.array([[2, 1],
              [1, -3],
              [1, 2]])

b = np.array([5, 10, -5]).transpose()

Im = np.array([1]*m)
In = np.array([1]*n)

# construct the problem(b)
x = cp.Variable(n)
cons = [cp.norm(x, 'inf') <= 1]
obj = cp.Minimize(cp.norm1(A @ x - b))
prob = cp.Problem(obj, cons)

# solve and output the result
prob.solve()
print('b)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var {}'.format(x.value))

s = cp.Variable(m)
cons = [-In <= x,
        x <= In,
        A @ x - b <= s,
        -s <= A @ x - b]
obj = cp.Minimize(Im @ s)
prob = cp.Problem(obj, cons)

prob.solve()
print('c)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x : {} s : {}'.format(x.value, s.value))
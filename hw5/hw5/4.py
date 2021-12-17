import cvxpy as cp
import numpy as np

X = np.array([[2, 0],
              [0, 1],
              [0, 0]])

y = np.array([3, 2, 2])

omega = cp.Variable(2)
'''(b)'''
# when t = 1
print('when t = 1')
cons = [cp.norm1(omega) <= 1]
obj = cp.Minimize(cp.norm2(X @ omega - y)**2)
prob = cp.Problem(obj, cons)

prob.solve()
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var {}'.format(omega.value))

# when t = 100
print('\nwhen t = 100')
cons = [cp.norm1(omega) <= 100]
prob = cp.Problem(obj, cons)

prob.solve()
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var {}'.format(omega.value))

'''(c)'''
# when t = 1
print('\nwhen t = 1')
cons = [cp.norm2(omega)**2 <= 1]
prob = cp.Problem(obj, cons)

prob.solve()
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var {}'.format(omega.value))

# when t = 100
print('\nwhen t = 100')
cons = [cp.norm2(omega)**2 <= 100]
prob = cp.Problem(obj, cons)

prob.solve()
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var {}'.format(omega.value))
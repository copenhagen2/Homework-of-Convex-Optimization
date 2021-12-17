import cvxpy as cp

# create two variables
x1 = cp.Variable()
x2 = cp.Variable()

# create the constraints of the problem
cons = [2 * x1 + x2 >= 1,
        x1 + 3 * x2 >= 1,
        x1 >= 0, x2 >= 0
        ]

#(a)
obj = cp.Minimize(x1 + x2)
prob = cp.Problem(obj, cons)
prob.solve()
print('a)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x1 : {} x2 : {}'.format(x1.value, x2.value))

#(b)
obj = cp.Minimize(- x1 - x2)
prob = cp.Problem(obj, cons)
prob.solve()
print('b)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x1 : {} x2 : {}'.format(x1.value, x2.value))

#(c)
obj = cp.Minimize(x1)
prob = cp.Problem(obj, cons)
prob.solve()
print('c)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x1 : {} x2 : {}'.format(x1.value, x2.value))

#(d)
obj = cp.Minimize(cp.maximum(x1, x2))
prob = cp.Problem(obj, cons)
prob.solve()
print('d)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x1 : {} x2 : {}'.format(x1.value, x2.value))

#(e)
obj = cp.Minimize(x1 ** 2 + 9 * x2 ** 2)
prob = cp.Problem(obj, cons)
prob.solve()
print('e)')
print('Status {}'.format(prob.status))
print('Optimal value {}'.format(prob.value))
print('Optimal var x1 : {} x2 : {}'.format(x1.value, x2.value))
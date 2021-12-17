# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2021-12-09 21:10:33
# @Last Modified by:   Your name
# @Last Modified time: 2021-12-09 22:57:07

import numpy as np
z = [1,2,1]
y = [1,1,-1]

def solve(z, y):
    n = len(y)
    p = m = 0
    for i in range(n):
        if y[i] > 0:
            p += 1
        else:
            m += 1
    if p == 0 or m == 0:
        return np.array([0] * n)
    u = sorted([ z[i] for i in range(n) if y[i] > 0])
    w = sorted([ -z[i] for i in range(n) if y[i] < 0])
    u = [-np.inf] + u + [np.inf]
    w = [-np.inf] + w + [np.inf]
    for k in range(0, p+1):
        for l in range(0, m+1):
            lambd = (sum(u[k+1:p+1]) + sum(w[1:l+1])) / (p - k + l)
            s1 = s2 = 0
            for i in u[1:p+1]:
                s1 += i - lambd if i-lambd > 0 else 0
            for i in w[1:m+1]:
                s2 += lambd - i if lambd-i > 0 else 0
            if abs(s1 - s2) <= 1e-5:
                return [z[i] - lambd*y[i] for i in range(n)], lambd

solution = solve(z, y)
print(f"optimal point : {solution[0]}")
print(f"lambda multiplier : {solution[1]}")
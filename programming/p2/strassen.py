# CS124 Programming 2 Assignment
# Helen Xiao and Elizabeth Zhong

import numpy as np
from random import randint
import time as time
import matplotlib.pyplot as plt

n_0 = 1

# helper matrix multiplication
def mat_mul(m1, m2):
    n = len(m1)
    result = np.array([[0]*n]*n)
    for r in range(n):
        for c in range(n):
            result[r][c] = np.dot(m1[r], m2[ :,c])
    return result


# strassen algorithm
def strassen(m1, m2):
    padded = False
    n = len(m1)
    if n <= n_0:
        return mat_mul(m1, m2)
    if n % 2 == 1:
        padded = True
        m1 = np.hstack((m1, np.zeros((n, 1), dtype=int)))
        m1 = np.vstack((m1, np.zeros((1, n+1), dtype=int)))
        m2 = np.hstack((m2, np.zeros((n, 1), dtype=int)))
        m2 = np.vstack((m2, np.zeros((1, n+1), dtype=int)))
        n += 1

    # split columns
    abcd = np.split(m1, 2, axis = 1)
    efgh = np.split(m2, 2, axis = 1)

    # split rows
    ac = np.split(abcd[0], 2, axis = 0)
    a = ac[0]
    c = ac[1]
    bd = np.split(abcd[1], 2, axis = 0)
    b = bd[0]
    d = bd[1]

    eg = np.split(efgh[0], 2, axis = 0)
    e = eg[0]
    g = eg[1]
    fh = np.split(efgh[1], 2, axis = 0)
    f = fh[0]
    h = fh[1]

    # calculate subproblems
    p1 = strassen(a, np.subtract(f,h))
    p2 = strassen(np.add(a,b), h)
    p3 = strassen(np.add(c,d), e)
    p4 = strassen(d, np.subtract(g,e))
    p5 = strassen(np.add(a,d), np.add(e,h))
    p6 = strassen(np.subtract(b,d), np.add(g,h))
    p7 = strassen(np.subtract(c,a), np.add(e,f))

    # calculate quadrants
    q1 = np.add(np.add(p5, p6), np.subtract(p4, p2))
    q2 = np.add(p1, p2)
    q3 = np.add(p3, p4)
    q4 = np.add(np.subtract(p1, p3), np.add(p5, p7))

    # splice together result
    result = np.vstack((np.hstack((q1,q2)), np.hstack((q3,q4))))
    if padded:
        result = np.split(result, [n-1], axis = 1)[0]
        result = np.split(result, [n-1], axis = 0)[0]

    # return result
    return result

# generate matrices
def gen_mat(n):
    result = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            result[r][c] = randint(0, 1)
    return result


# Trials
def run_trials(n):
    count = [0]*n
    strassen_times = [0]*n
    mat_mul_times = [0]*n
    for i in range(n):
        print(i)
        count[i] = i
        m1 = gen_mat(i)
        m2 = gen_mat(i)

        # time strassen
        start = time.time()
        strassen(m1, m2)
        end = time.time()
        strassen_times[i] = end - start

        # time mat_mul
        start = time.time()
        mat_mul(m1, m2)
        end = time.time()
        mat_mul_times[i] = end - start

    plt.plot(count, strassen_times, label = "Strassen Runtime")
    plt.plot(count, mat_mul_times, label = "Mat_Mul Runtime")
    plt.legend
    plt.show()

run_trials(50)


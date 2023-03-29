#...............................................................................
#                         CS124 Programming 2 Assignment
#                         Helen Xiao and Elizabeth Zhong
#...............................................................................
import sys
import numpy as np
#from random import randint, uniform
#import time as time
#import matplotlib.pyplot as plt

#...............................................................................
#                             Part 2 - Strassen Trials
#...............................................................................

# Conventional Matrix Multiplication
def mat_mul(A, B):
    n = len(A)
    C = np.array([[0 for i in range(n)] for j in range(n)])
    for i in range(n):
        for j in range(n):
            C[i][j] += np.dot(A[i], B[:,j])
    return C
"""
def mat_mul(m1, m2):
    return np.dot(m1, m2)
"""

# Strassen's Algorithm
def strassen(m1, m2, n_0):
    padded = False
    n = len(m1)
    # Check against threshold
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
    p1 = strassen(a, np.subtract(f,h), n_0)
    p2 = strassen(np.add(a,b), h, n_0)
    p3 = strassen(np.add(c,d), e, n_0)
    p4 = strassen(d, np.subtract(g,e), n_0)
    p5 = strassen(np.add(a,d), np.add(e,h), n_0)
    p6 = strassen(np.subtract(b,d), np.add(g,h), n_0)
    p7 = strassen(np.subtract(c,a), np.add(e,f), n_0)
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
"""
# Binary Matrix Generator
def gen_mat(n):
    result = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            result[r][c] = randint(0, 1)
    return result

# Run and Plot Trials
def run_trials(n):
    # Generate matrices
    M1 = gen_mat(n)
    M2 = gen_mat(n)

    # Run trials
    num_tests = 5
    k_times = []
    for k in range(30, n):
        times = []
        for i in range(num_tests):
            start = time.time()
            strassen(M1, M2, k)
            end = time.time()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        print("k =", k, "Average time taken:", avg_time)
        k_times.append((avg_time, k))

    # Print k with smallest runtime
    print(min(k_times)) 

    # Plot
    x = [kt[1] for kt in k_times]
    y = [kt[0] for kt in k_times]
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("Average time taken (s)")
    plt.title("Optimal k for Strassen")
    plt.show()
"""
#...............................................................................
#                                Part 3 - Triangles
#...............................................................................
"""
# random bernoulli generator
def bern(p):
    if uniform(0, 1) < p:
        return 1
    else: return 0

# Undirected Adjacency Matrix Generator
def gen_adj_mat(n, p):
    result = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(r+1, n):
            result[r][c] = bern(p)
            result[c][r] = result[r][c]
    return result

# Triangle Calculator
def triangle(a):
    n = len(a)
    a_3 = strassen(strassen(a, a, 80), a, 80)
    num = 0
    for i in range(n):
        num += a_3[i][i]
    return num // 6

num_trials = 10

def run_triangle(n):
    ps = [0.01, 0.02, 0.03, 0.04, 0.05]
    for p in ps:
        avg_tri = 0
        for i in range(num_trials):
            m = gen_adj_mat(n, p)
            avg_tri += triangle(m)
        avg_tri /= num_trials
        print("n =", n, "p =", p, "Average Triangle:", avg_tri)
"""
#...............................................................................
#                                      Main
#...............................................................................
def main():
    n = len(sys.argv)
    if n != 4:
        sys.exit("Usage: python3 strassen.py 0 dimension inputfile")
    d = int(sys.argv[2].strip())
    m1 = np.zeros((d, d), dtype=int)
    m2 = np.zeros((d, d), dtype=int)
    input = open(sys.argv[3], "r")
    lines = input.readlines()

    line = 0
    for i in range(d):
        for j in range(d):
            m1[i][j] = int(lines[line].strip())
            line += 1

    for i in range(d):
        for j in range(d):
            m2[i][j] = int(lines[line].strip())
            line += 1

    C = strassen(m1, m2, 100)
    for i in range(d):
        print(C[i][i])

main()
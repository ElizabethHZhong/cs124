import numpy as np
import time

# recursive definition
def rec_fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return rec_fib(n-1) + rec_fib(n-2)

# iterative definition
def iter_fib(n):
    fibs = [0,1]
    for i in range(2, n + 1):
        fibs.append(fibs[i-1] + fibs[i-2])
    return fibs[n]

# matrix definition
def matrix_fib(n):
    start = np.array([[1],[0]])
    fib = np.array([[1,1], [1,0]])
    fib = np.linalg.matrix_power(fib, n-1)
    return np.matmul(fib, start)[0][0]

# testing function
def testing(n):
    print("Recursive: " + str(rec_fib(n)))
    print("Iterative: " + str(iter_fib(n)))
    print("Matrix: " + str(matrix_fib(n)))

def timer(n):
    start = time.time()
    rec_fib(n)
    end = time.time()
    print("Recursive: " + str(end - start))
    start = time.time()
    iter_fib(n)
    end = time.time()
    print("Iterative: " + str(end - start))
    start = time.time()
    matrix_fib(n)
    end = time.time()
    print("Matrix: " + str(end - start))

def overflow():
    fibs = [0,1]
    #count = 2
    for count in range(10):
        fibs.append(fibs[count-1] + fibs[count-2])
        print(fibs[count])
        #count += 1
    
overflow()

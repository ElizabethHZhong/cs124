import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
    return int(np.matmul(fib, start)[0][0])

# testing function
def testing(n):
    print("Recursive: " + str(rec_fib(n)))
    print("Iterative: " + str(iter_fib(n)))
    print("Matrix: " + str(matrix_fib(n)))

def timer(n):
    print(str(n) + " numbers")

    # matrix timing
    start = time.time()
    matrix_fib(n)
    end = time.time()
    print("Matrix: " + str(end - start))

    # iterative
    start = time.time()
    iter_fib(n)
    end = time.time()
    print("Iterative: " + str(end - start))
    
    # recursive
    #start = time.time()
    #rec_fib(n)
    #end = time.time()
    #print("Recursive: " + str(end - start))

    

def overflow():
    fibs = [0,1]
    for count in range(2, 50):
        fibs.append(fibs[count-1] + fibs[count-2])
        print(count, end=": ")
        print(fibs[count])

# 2c
# 2^16
upper_bound = 65536 

# mod recursive definition
def mod_rec_fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return (mod_rec_fib(n-1) + mod_rec_fib(n-2)) % upper_bound

# mod iterative definition
def mod_iter_fib(n):
    if n == 0:
        return 0
    a = 0
    b = 1
    curr = a + b
    for i in range(3, n + 1):
        a = b
        b = curr
        curr = (a + b) % upper_bound
    return curr

# mod matrix definition
def mod_matrix_fib(n):
    if n == 0 or n == 1:
        return n
    start = np.array([[1],[0]])
    fib = np.array([[1,1], [1,0]])
    pow_fib = np.array([[1,0], [0,1]]) # accumulator
    
    for j in range(n-1):
        pow_fib = np.matmul(fib, pow_fib)
        for r in range(2):
            for c in range(2):
                pow_fib[r][c] = pow_fib[r][c] % upper_bound
    return np.matmul(pow_fib, start)[0][0]

def fibonaccim(n, mod=False):
    fmat = np.array([[1,1],[1,0]])
    first_two = np.array([1,0])
    res = np.matmul(np.linalg.matrix_power(fmat,n),first_two)
    if mod:
        return res[0] % pow(2,16)
    return res[1]

def timing(n):
    print("start")
    start = time.time()
    mod_matrix_fib(n)
    end = time.time()
    print(end - start)

#timing(2**(300000))

def one_min():
    timeout = time.time() + 60
    count = 2
    a = 0
    b = 1
    curr = a + b
    while time.time() < timeout:
        a = b
        b = curr
        curr = (a + b) % upper_bound
        count += 1
        if time.time() >= timeout:
            break
    print("Index: ", end="")
    print(count)

# testing function
def testing(n):
    print("Recursive: " + str(rec_fib(n)))
    print("Iterative: " + str(iter_fib(n)))
    print("Matrix: " + str(matrix_fib(n)))

    
def time_graph(nmax, times_to_run = 10):
    times = pd.DataFrame(columns=["n", "Fib", "Time", "Type"])
    functions=[iter_fib, matrix_fib]
    for i in range(nmax+1):
        for function in functions:
            times_to_avg = []
            for _ in range(times_to_run):
                start = time.time()
                function(i)
                elapsed = time.time() - start
                times_to_avg.append(elapsed)
            row_dict = {"n" : [i], 
                        "Fib" : [function(i)],
                        "Time" : [np.mean(times_to_avg)],
                        "Type" : [function.__name__]}
            row = pd.DataFrame.from_dict(row_dict)
            times = pd.concat([times, row])
    return times.reset_index().drop("index",axis=1)

def plot(times):
    sns.lineplot(data=times, x="n", y="Time", hue="Type")
    plt.show()


plot(time_graph(10**3))
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    start = time.time()
    rec_fib(n)
    end = time.time()
    print("Recursive: " + str(end - start))
    

def overflow():
    fibs = [0,1]
    for count in range(2, 50):
        fibs.append(fibs[count-1] + fibs[count-2])
        print(count, end=": ")
        print(fibs[count])

# 2^16
upper_bound = 65536 

#def one_min():
    #timeout = time.time() + 60
    #while time.time() < timeout:
        #if n == 0 or n == 1:
            #return n
        #else:
            #return (rec_fib(n-1) + rec_fib(n-2)) mod upper_bound
    
def time_graph(nmax, times_to_run = 10):
    times = pd.DataFrame(columns=["n", "Fib", "Time", "Type"])
    functions=[rec_fib, iter_fib, matrix_fib]
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
    seaborn.lineplot(data=times, x="n", y="Time", hue="Type")
    plt.show()

plot(time_graph(20))

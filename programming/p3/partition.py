#...............................................................................
#                         CS124 Programming 3 Assignment
#                        Rhea Acharya and Elizabeth Zhong
#...............................................................................
import sys
from heapq import heappop, heappush, heapify
from random import randint, uniform, choice
from math import exp, floor
import time as time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#...............................................................................
#                             Karmarkar-Karp Algorithm
#...............................................................................

def kk(a):
    if len(a) == 0:
        return -1
    h = []
    heapify(h)
    n = len(a)
    for i in range(n):
        heappush(h, -1 * a[i])
    while(len(h) > 1):
        m1 = -1 * heappop(h)
        m2 = -1 * heappop(h)
        d = m1 - m2
        heappush(h, -d)
    return -1 * heappop(h)

#...............................................................................
#                             Repeated Random Algorithm
#...............................................................................

def get_random(n):
    return [choice([-1, 1]) for i in range(n)]

def partition(s, a):
    n = len(s)
    h1 = []
    h2 = []
    for i in range(n):
        if s[i] == 1:
            h1.append(a[i])
        else:
            h2.append(a[i])
    return h1, h2

def res(h):
    h1, h2 = h
    return abs(sum(h1) - sum(h2))

def repeated(a, m):
    n = len(a)
    curr_res = res(partition(get_random(n), a))
    for i in range(m):
        next_res = res(partition(get_random(n), a))
        if next_res < curr_res:
            curr_res = next_res
    return curr_res

#...............................................................................
#                             Hill Climbing Algorithm
#...............................................................................

def rand_neighbor(s):
    n = len(s)
    rand_ind = randint(0, n-1)
    s[rand_ind] *= -1
    return s

def hill(a, m):
    n = len(a)
    curr = get_random(n)
    curr_res = res(partition(curr, a))
    for i in range(m):
        next = rand_neighbor(curr)
        next_res = res(partition(next, a))
        if next_res < curr_res:
            curr_res = next_res
            curr = next
    return curr_res

#...............................................................................
#                           Simulated Annealing Algorithm
#...............................................................................

# random bernoulli generator
def bern(p):
    if uniform(0, 1) < p:
        return 1
    else: return 0

def t(iter):
    return (10 ** 10) * (0.8 ** floor(iter / 300)) 

def annealing(a, m):
    n = len(a)
    curr = get_random(n)
    curr_res = res(partition(curr, a))
    result = curr
    result_res = res(partition(result, a))
    for i in range(m):
        next = rand_neighbor(curr)
        next_res = res(partition(next, a))
        if next_res < curr_res:
            curr_res = next_res
            curr = next
        else:
            p = exp((- next_res + curr_res) / t(i))
            if bern(p) == 1:
                curr_res = next_res
                curr = next
        if curr_res < result_res:
            result = curr
            result_res = curr_res
        
    return result_res

#...............................................................................
#                               Preparition Algorithm
#...............................................................................

def gen_rand(n):
    return [randint(1, n) for i in range(n)]

def prepartition(p, a):
    n = len(a)
    result = [0] * n
    for i in range(n):
        result[p[i]-1] += a[i]
    return result
        

#...............................................................................
#                           Algorithms with Prepartitioning
#...............................................................................

def pre_repeated(a, m):
    n = len(a)
    curr_res = kk(prepartition(gen_rand(n), a))
    for i in range(m):
        next_res = kk(prepartition(gen_rand(n), a))
        if next_res < curr_res:
            curr_res = next_res
    return curr_res


# pre-neighbor
def pre_rand_neighbor(s):
    n = len(s)
    rand_ind = randint(0, n-1)
    s[rand_ind] = randint(1, n)
    return s

def pre_hill(a, m):
    n = len(a)
    curr = gen_rand(n)
    curr_res = kk(prepartition(curr, a))
    for i in range(m):
        next = pre_rand_neighbor(curr)
        next_res = kk(prepartition(next, a))
        if next_res < curr_res:
            curr_res = next_res
            curr = next
    return curr_res


def pre_annealing(a, m):
    n = len(a)
    curr = gen_rand(n)
    curr_res = kk(prepartition(curr, a))
    result = curr
    result_res = kk(prepartition(result, a))
    for i in range(m):
        next = pre_rand_neighbor(curr)
        next_res = kk(prepartition(next, a))
        if next_res < curr_res:
            curr_res = next_res
            curr = next
        else:
            p = exp((- next_res + curr_res) / t(i))
            if bern(p) == 1:
                curr_res = next_res
                curr = next
        if curr_res < result_res:
            result = curr
            result_res = curr_res
    return result_res

#...............................................................................
#                                50 Random Instances
#...............................................................................

MAX_ITER = 25000
max_num = 10 ** 12

def gen_a(n):
    return [randint(1, max_num) for i in range(n)]

def graph(functions, n, l, arr):
    for f in functions:
        print("Function", f.__name__)
        d = pd.DataFrame(columns=["Residue"])
        for i in range(n):
            print("Instance", i)
            a = arr[i]
            row_dict = {"Residue" : [f(a, MAX_ITER)]}
            row = pd.DataFrame.from_dict(row_dict)
            d = pd.concat([d, row])
        sns.histplot(data=d, x="Residue")
        plt.title("Residue Histogram for " + str(f.__name__))
        plt.show()

def graph_scatter(functions, n, arr):
    d = pd.DataFrame(columns=["Residue", "Function"])
    for i in range(n):
        print("Instance", i)
        for f in functions:
            a = arr[i]
            res = f(a, MAX_ITER)
            print("Function", f.__name__, res)
            row_dict = {"Residue" : [res],
                        "Function" : [f.__name__]}
            row = pd.DataFrame.from_dict(row_dict)
            d = pd.concat([d, row])
        print("KK")
        a = arr[i]
        row_dict = {"Residue" : [kk(a)],
                    "Function" : ["KK"]}
        row = pd.DataFrame.from_dict(row_dict)
        d = pd.concat([d, row])
    #d.to_csv('timing-data.csv')
    fig, ax = plt.subplots()
    sns.stripplot(data=d, x="Function", y="Residue", hue="Function", ax=ax)
    fig.set_figwidth(12)
    fig.set_figheight(6)
    ax.set_yscale('log')
    ax.set_title("Residue Scatterplot")
    plt.savefig('egg.png')

def graph_kk(n, l, arr):
    print("Function Karmarkar-Karp")
    d = pd.DataFrame(columns=["Residue"])
    for i in range(n):
        print("Instance", i)
        a = arr[i]
        row_dict = {"Residue" : [kk(a)]}
        row = pd.DataFrame.from_dict(row_dict)
        d = pd.concat([d, row])
    sns.histplot(data=d, x="Residue")
    plt.title("Residue Histogram for Karmarkar-Karp")
    plt.show()

def graph_time(functions, trials, nmax):
    times = pd.DataFrame(columns=["n", "Time", "Function"])
    for j in range(1, nmax+1):
        n = 2 ** j
        print(n)
        arrs = [gen_a(n) for _ in range(trials)]
        for f in functions:
            print(f.__name__)
            times_to_avg = []
            for i in range(trials):
                start = time.time()
                f(arrs[i], MAX_ITER)
                elapsed = time.time() - start
                times_to_avg.append(elapsed)
            row_dict = {"n" : [n],
                        "Time" : [np.mean(times_to_avg)],
                        "Function" : [f.__name__]}
            row = pd.DataFrame.from_dict(row_dict)
            times = pd.concat([times, row])
        print("KK")
        times_to_avg = []
        for i in range(trials):
            start = time.time()
            kk(arrs[i])
            elapsed = time.time() - start
            times_to_avg.append(elapsed)
        row_dict = {"n" : [n],
                    "Time" : [np.mean(times_to_avg)],
                    "Function" : [kk.__name__]}
        row = pd.DataFrame.from_dict(row_dict)
        times = pd.concat([times, row])
    sns.lineplot(data=times, x="n", y="Time", hue="Function")
    plt.title("Average Runtime per Function")
    plt.show()
    


#...............................................................................
#                                     Main
#...............................................................................

def main():
    n = len(sys.argv)
    if n != 4:
        sys.exit("Usage: python3 partition.py flag algorithm inputfile")
    flag = int(sys.argv[1].strip())
    alg = int(sys.argv[2].strip())
    input = open(sys.argv[3], "r")
    lines = input.readlines()

    max_iter = 25000
    n = 100
    a = [0] * n

    for i in range(n):
        a[i] = int(lines[i].strip())

    instances = 50
    arr_size = 100
    arrs = [gen_a(arr_size) for i in range(instances)]
    functions = [repeated, hill, annealing, 
                 pre_repeated, pre_hill, pre_annealing]
    trials = 25
    nmax = 7

    if flag == 0:
        if alg == 0:
            print(kk(a))
        elif alg == 1:
            print(repeated(a, max_iter))
        elif alg == 2:
            print(hill(a, max_iter))
        elif alg == 3:
            print(annealing(a, max_iter))
        elif alg == 11:
            print(pre_repeated(a, max_iter))
        elif alg == 12:
            print(pre_hill(a, max_iter))
        elif alg == 13:
            print(pre_annealing(a, max_iter))
        else:
            print("Not a valid algorithm")
    elif flag == 1:
        graph_kk(instances, arr_size, arrs)
        graph(functions, instances, arr_size, arrs)
    elif flag == 2:
        graph_time(functions, trials, nmax)
    elif flag == 3:
        graph_scatter(functions, instances, arrs)


main()
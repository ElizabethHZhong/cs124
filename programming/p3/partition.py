#...............................................................................
#                         CS124 Programming 3 Assignment
#                        Rhea Acharya and Elizabeth Zhong
#...............................................................................
import sys
from heapq import heappop, heappush, heapify
from random import randint, uniform, choice
from math import exp, floor

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



#...............................................................................
#                                     Main
#...............................................................................

def main():
    n = len(sys.argv)
    if n != 4:
        sys.exit("Usage: python3 partition.py flag algorithm inputfile")
    alg = int(sys.argv[2].strip())
    input = open(sys.argv[3], "r")
    lines = input.readlines()

    max_iter = 25000
    n = 100
    a = [0] * n

    for i in range(n):
        a[i] = int(lines[i].strip())

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

main()
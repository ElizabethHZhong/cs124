from decimal import Decimal
from math import comb

def fpp_p(k, n):
    m = 8 * n
    nk = n * k
    p = Decimal(1 / m)
    q = Decimal(1 - p)
    sum = 0
    for j in range(16):
        bin = comb(nk, j)
        sum += Decimal(bin * (p ** j) * (q ** (nk - j)))
    fpp = 1 - sum
    print("k:", k, "n:", n, "m:", m)
    print("FPP:", fpp, "Expected:", m * fpp)

def fpp_n(k, n):
    m = 8 * n
    p = Decimal(k / m)
    q = Decimal(1 - p)
    sum = 0
    for j in range(16):
        bin = comb(n, j)
        sum += Decimal(bin * (p ** j) * (q ** (n - j)))
    fpp = 1 - sum
    print("k:", k, "n:", n, "m:", m)
    print("FPP:", fpp, "Expected:", m * fpp)

fpp_p(5, 1000)
fpp_p(5, 10000)
fpp_p(5, 100000)

p = 1249
P = [3,1,4,1,5,9,2,6]
D = [1,4,6,8,3,4,8,3,1,9,4,1,6,7,8,3,9,3,3,1,4,1,7,1,7,5,1,1,3,2]

def arr_to_int(a, k):
    result = 0
    exp = 10 ** (k - 1)
    for i in range(k):
        result += a[i] * exp
        exp //= 10
    return result

def hashed(a, k, p):
    return arr_to_int(a, k) % p

def compare(p, P, D):
    count = 1
    k = len(P)
    n = len(D)
    hash = [0]*(n-k+1)
    for i in range(n-k+1):
        hash[i] = hashed(D[i:i+k], k, p)
    hashed_p = hashed(P, k, p)
    for i in range(n-k+1):
        if hashed_p == hash[i]:
            print(P, "vs.", D[i: i+k])
            found = True
            for j in range(k):  
                curr_p = P[j]
                curr_d = D[i + j]
                print(count, "Comparing P[", j, "]:", curr_p, "and D[", j + i, "]:", curr_d)
                count += 1
                if curr_p != curr_d:
                    print("Not a match")
                    print()
                    found = False
                    break
                else:
                    print("Match!")
            if found == True:
                print("Full match!")
                break

#compare(p, P, D)




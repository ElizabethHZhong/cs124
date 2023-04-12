def primality(n):
    a = 1
    m = 1
    while(m == 1):
        a += 1
        m = pow(a, n-1, n)
    return a

print(primality(12403180369))

def carmichael(n):
    for a in range(n):
        if (a % n != 1) and ((a ** 2) % n == 1):
            return a

print(carmichael(63973))
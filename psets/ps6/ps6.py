import math

def primality(n):
    a = 1
    m = 1
    while(m == 1):
        a += 1
        m = pow(a, n-1, n)
    return a

#print(primality(12403180369))

def carmichael(n):
    for a in range(n):
        if (a % n != 1) and ((a ** 2) % n == 1):
            return a

print(carmichael(63973))
n = 63973

def decrypt(n):
    upper = math.ceil(math.sqrt(n))
    for i in range(upper, 0, -1):
        if n % i == 0:
            return (i, n // i)

print(decrypt(15375998174720047661999))


print("6252 mod", n, "= ", 6252 % n)
a = 6252 ** 2
print(a, "mod", n, "= ", a % n)
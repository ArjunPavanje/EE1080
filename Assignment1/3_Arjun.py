import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
random.seed(42)

# Generating U (an n x N matrix filled with random values between 0, 1)
def Ugen(n, N):
    U = [[random.uniform(0, 1) for _ in range(N)] for _ in range(n)]
    return U 

def Igen(U, k, n, N):
    I = np.zeros_like(U)
    col_sum = np.zeros(N) # Contains sum of elements in each column of I
    
    # First column of I 
    for i in range (N):
        if U[0][i] <= k/n: # Applying formula given in the question
            I[0][i] = 1
            col_sum[i] += 1 # incrementing column sum 
    
    # Remaining Columns

    for i in range(1, n):
        for j in range(N):
            if U[i][j] <= (k - col_sum[j])/(n-i): # Applying formula given in the question
                I[i][j] += 1
                col_sum[j] += 1 # incrementing column sum
    
    return I


# Takes I matrix whose columns are binary numbers and returns an array with equivalent decimal representation
def bin_to_dec(I, n, N):
    y = np.zeros(N)

    for i in range(N):
        col = I[:, i]
        num = 0
        for j in range(n):
            num += (2**j)*col[j]
        y[i] = num

    return y

# Error Handling
if len(sys.argv) != 4:
    sys.exit("Invalid Number of Parameters")

# Accepting sys args
n, k, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# Error Handling
if n < 0 or N < 0:
    sys.exit("n and N have to be non negative integers")
if n < k:
    sys.exit("k has to be lesser than n")

# Generating U, I, output matrices/arrays
U = Ugen(n, N)
I = Igen(U, k, n, N)
y = bin_to_dec(I, n, N)

# Plotting Histogram
plt.hist(y, bins = 2**n, range = (0, 2**n))
plt.title(f'Generating equally likely {k} subsets of {n}')
plt.xlabel('Decimal')
plt.ylabel('Frequency')
plt.show()


import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
random.seed(42)

def generate_samples(mode, n, N):
    Y = np.zeros(N)
    samples = np.zeros((n, N))
    if mode == 0:
        p = float(sys.argv[4])
        samples = np.random.binomial(1, p, size=(n, N))
    elif mode == 1:
        samples = np.random.uniform(0, 1, size=(n, N))
    elif mode == 2:
        lamb = float(sys.argv[4])
        samples = np.random.exponential(1/lamb, size=(n, N))


    for i in range (N):
        for j in samples[:, i]:
            Y[i]+=j
        Y[i]/=n

    return Y, samples

def plot(Y, n, N, samples, mode):
    x = np.linspace(min(Y), max(Y), 10000)
    mean = np.mean(samples)
    variance = np.var(samples)/n
    '''
    mean = 0 
    sum_squared = 0
    for i in range (n):
        for j in range (N):
            mean += samples[i][j] 
            sum_squared += samples[i][j]**2
    mean/=(n*N)
    sum_squared/=(n*N)
    variance = sum_squared - (mean**2)'''
    y = np.exp(-0.5*((x-mean)**2/variance))/(np.sqrt(variance*2*np.pi))
    plt.plot(x, y, label='Normal PDF')
    if mode == 0:
        plt.hist(Y, density = True, alpha = 0.5, label = 'Histogram')
    else:
        plt.hist(Y, bins = int(np.sqrt(N)), density = True, alpha = 0.5, label = 'Histogram')

    plt.legend()
    plt.show()


'''
# Error Handling
if (len(sys.argv) != 3) or (len(sys.argv) != 4):
    sys.exit("Invalid Number of Parameters")
'''

# Accepting sys args
mode, n, N =int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

Y, samples = generate_samples(mode, n, N)
plot(Y, n, N, samples, mode)

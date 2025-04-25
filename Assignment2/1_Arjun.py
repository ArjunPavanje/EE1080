import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
np.random.seed(1)

def generate_samples(mode, n, N):

    # Generating Samples of RV's based on mode
    Y = np.zeros(N)
    samples = np.zeros((n, N))
    if mode == 0:
        p = float(sys.argv[4])
        samples = np.random.binomial(1, p, size=(N, n))
    elif mode == 1:
        samples = np.random.uniform(0, 1, size=(N, n))
    elif mode == 2:
        lamb = float(sys.argv[4])
        samples = np.random.exponential(1/lamb, size=(N, n))

    # calculating average of each row
    for i in range (N):
        for j in samples[i, :]:
            Y[i]+=j
        Y[i]/=n

    return Y, samples

# Plotting histogram and Normal PDF
def plot(Y, n, N, samples, mode):

    # Calculating mean and variance for Normal RV
    x = np.linspace(min(Y), max(Y), 10000)
    mean = 0 
    if mode == 0:
        p = float(sys.argv[4])
        mean = p 
        variance = p*(1-p)/n
    elif mode == 1:
        mean = 0.5
        variance = 1/(12*n)
    elif mode == 2:
        lamb = float(sys.argv[4])
        mean = 1/lamb 
        variance = 1/((lamb**2) * n)
    y = np.exp(-0.5*((x-mean)**2/variance))/(np.sqrt(variance*2*np.pi))

    # Plot and related Settings
    plt.plot(x, y, label='Normal PDF')
    plt.hist(Y, bins = "auto", density = True, alpha = 0.5, label = 'Histogram')
    plt.xlabel("Samples")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Accepting sys args
mode, n, N =int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# Calling Functions
Y, samples = generate_samples(mode, n, N)
plot(Y, n, N, samples, mode)

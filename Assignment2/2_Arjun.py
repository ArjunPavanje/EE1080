import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
random.seed(42)

def generate_X(N):
    X = np.zeros((N, 6))
    for i in range (N):
        arrival_times = np.random.uniform(0, 1, 6)
        arrival_times.sort()
        #X[i] = arrival_times[3]
        X[i, :] = arrival_times
    time = X[:, 3]
    return time

def plot(X, N):
    x = np.linspace(0, 1, 1000)
    fa = 60*np.pow(x, 3)*np.pow((1-x), 2)
    fb = 30*np.pow(x, 4)*np.pow((1-x), 1)

    hist_vals, bin_edges, _ = plt.hist(X, bins = 100, density = True, alpha = 0.5, label = 'Histogram')
    plt.plot(x, fa, label = 'fa')
    plt.plot(x, fb, label = 'fb')
    plt.legend()
    plt.show()

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Evaluate PDFs at bin centers
    fa_bins = 60 * np.power(bin_centers, 3) * np.power((1 - bin_centers), 2)
    fb_bins = 30 * np.power(bin_centers, 4) * np.power((1 - bin_centers), 1)

    # Compute mean squared errors
    mse_fa = np.mean((hist_vals - fa_bins) ** 2)
    mse_fb = np.mean((hist_vals - fb_bins) ** 2)

    # Print result based on closer match
    if mse_fa < mse_fb:
        print('a')
    else:
        print('b')


'''
# Error Handling
if (len(sys.argv) != 3) or (len(sys.argv) != 4):
    sys.exit("Invalid Number of Parameters")
'''

# Accepting sys args
N =int(sys.argv[1])
X = generate_X(N)
plot(X, N)

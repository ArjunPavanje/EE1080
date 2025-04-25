import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
np.random.seed(69)

# Generating uniform random variable samples, and returning array containing time at which game started for each of the 'N' games
def generate_X(N):
    X = np.random.uniform(0, 1, size = (N, 6))
    for i in range(N):
        X[i] = np.sort(X[i])

    time = X[:, 3]
    return time

def plot(X, N):
    # Defining fa, fb
    x = np.linspace(0, 1, 1000)
    fa = 60*(x**3)*((1-x)**2)
    fb = 30*(x**4)*(1-x)

    # Plot and related settings
    bin_edges = np.arange(0, 1+0.01, 0.01)
    hist_vals, bin_edges, _ = plt.hist(X, bins = 100, density = True, alpha = 0.5, label = 'Histogram')
    plt.plot(x, fa, label = 'fa')
    plt.plot(x, fb, label = 'fb')
    plt.legend()
    plt.show()
    
    # Checking if histogram plot is closer to fa or fb by calculating mean square error

    # Computing bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Evaluate PDFs at bin centers
    fa_bins = 60 * np.power(bin_centers, 3) * np.power((1 - bin_centers), 2)
    fb_bins = 30 * np.power(bin_centers, 4) * np.power((1 - bin_centers), 1)

    # Compute mean squared errors
    mse_fa = np.mean((hist_vals - fa_bins) ** 2)
    mse_fb = np.mean((hist_vals - fb_bins) ** 2)

    # Print result based on which curve it matches 
    if mse_fa < mse_fb:
        print('a')
    else:
        print(fb)


'''
# Error Handling
if (len(sys.argv) != 3) or (len(sys.argv) != 4):
    sys.exit("Invalid Number of Parameters")
'''

# Accepting sys args
N =int(sys.argv[1])
X = generate_X(N)
plot(X, N)

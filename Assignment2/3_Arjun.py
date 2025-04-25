import random
import math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
np.random.seed(69)

'''
We get x^_{MMSE} after simplification to be,
                          mu_0/var_0 + summation (i = 0 to i) y_i / var_i
X^[Y_1, Y_2, ..., Y_N] = -------------------------------------------------
                            1/var_0 + summation (i = 0 to i) 1/var_i
'''

# Calculating MMSE according to above mentioned formula
def mmse_calculator(mu_0, var_0, y_i, var_i):
    mmse_estimates = []
    for n in range(1, len(y_i) + 1):
        y = y_i[:n]
        var = var_i[:n]

        numerator = mu_0 / var_0 + sum(y[i] / var[i] for i in range(n))
        denominator = 1 / var_0 + sum(1 / var[i] for i in range(n))

        mmse_estimate = numerator / denominator

        mmse_estimates.append(mmse_estimate)
    return mmse_estimates

# Plotting and related settings
def plot(mmse_estimates):
    x = np.arange(1, len(mmse_estimates) +1)
    plt.scatter(x, mmse_estimates, label='Convergence of X^', color = "chartreuse")
    plt.xlabel("MMSE Estimates")
    plt.ylabel("MMSE of X")
    plt.legend()
    plt.show()

def process_csv(file_path):
    # Read and display the CSV file
    try:
        df = pd.read_csv(file_path, skiprows=0)
        print("Reading contents of ",  file_path, ":\n")
        return df;
    except Exception as e:
        print(f"Error reading CSV file: {e}")

if __name__ == "__main__":        
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])

    print("\nArguments passed:",n)
    assert(n >= 2)

    mu_x = int(sys.argv[1]);
    print("mean of X", mu_x)

    var_x = int(sys.argv[2]);
    print("Variance of X", var_x)

    #read the mmse samples file
    input_fn= sys.argv[3];
    print(input_fn)
    mmse_samples = process_csv(input_fn);
    print(mmse_samples)
    N = len(mmse_samples);
    print("N=",N);
    mmse_samples_array = mmse_samples.to_numpy();
    print(mmse_samples_array)
    # prints below 0th sample value
    print(mmse_samples_array[0][0])
    #prints below sigmasquare corresponding to 0th sample
    print(mmse_samples_array[0][1])
    #similarly parse the third argument if mode is 0 or 1 to get p or lambda

    y_i = mmse_samples_array[:, 0]
    var_i = mmse_samples_array[:, 1]

    mmse_estimates = mmse_calculator(mu_x, var_x, y_i, var_i)
    plot(mmse_estimates)


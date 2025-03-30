import sys
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


# Returns Bernouli Sample from Uniform Sample 
def bernoulli(x, p):
    if 0 < x < p: 
        return 1
    else:
        return 0
# For .map (to avoid lambda)
def bernoulli_func(x):
    return bernoulli(x, p)

# Returns Exponential Sample from Uniform Sample 
def exponential(x, lamb):
    if x == 1: 
        return 0
    return -np.log(1 - x)/lamb
# For .map (to avoid lambda)
def exponential_func(x):
    return exponential(x, lamb)

# Generating Random Variables from the CDF given in the question 
def custom_cdfx(x):
    if 0 <= x <= 1/3:
        return np.sqrt(3 * x)
    elif 2/3 <= x <= 1:
        return (6*x) - 2
    else:
        return 2  



def process_csv(file_path):
    # Read and display the CSV file
    try:
        df = pd.read_csv(file_path, skiprows=0)
        print("Reading contents of ",  file_path, ":\n")
        return df;
    except Exception as e:
        print("Error reading CSV file: {e}")

if __name__ == "__main__":        
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])

    print("\nArguments passed:",n)
    assert(n >= 2)

    mode_value = int(sys.argv[1]);
    print("Testing mode:", mode_value)

    #read the samples file
    input_fn= sys.argv[2];
    uniform_samples = process_csv(input_fn);
    print(uniform_samples)
    N = len(uniform_samples);
    print("N=",N);
    #similarly parse the third argument if mode is 0 or 1 to get p or lambda 



    if mode_value == 0:
        assert n >= 3 # Error Handling
        
        # Accepting sys args
        p = float(sys.argv[3])

        # Error Handling
        if p>1 or p<0:
            sys.exit("p should be between 0, 1")

        peas = str(p).strip("0").replace(".", "p")  # For naming CSV
        bernoulli_samples = uniform_samples.map(bernoulli_func)
        bernoulli_samples.rename(columns={'Uniform Samples': 'Bernoulli Samples'}, inplace=True) # Mapping uniform_samples to binmial samples generated from it
        bernoulli_samples.to_csv(f"Bernoulli_{peas}.csv") # Writing to CSV
        bernoulli_mean = round(bernoulli_samples.mean()['Bernoulli Samples'], 3) # Calculating mean using a pandas function
        print(bernoulli_mean)

    elif mode_value == 1:
        assert n >= 3 # Error Handling 

        # Read the lambda value from sys args
        lamb = float(sys.argv[3])

        # Error Handling
        if lamb < 0:
            sys.exit("lambda should be non negative")
        
        leas = str(lamb).strip("0").replace(".", "p")  # For naming csv
        exponential_samples = uniform_samples.map(exponential_func) # # Mapping uniform_samples to exponential samples generated from it
        exponential_samples.rename(columns={'Uniform Samples': 'Exponential Samples'}, inplace=True) # Creating two columns
        exponential_samples.to_csv(f"Exponential_{leas}.csv") # Writing to CSV

        # Plotting Histogram
        plt.hist(exponential_samples, bins=int(np.sqrt(N))) 
        plt.title(f"Exponential Samples with lambda = {lamb}")
        plt.xlabel("Frequency")
        plt.ylabel("Exponential Samples")
        plt.show()

    elif mode_value == 2:

        cdfx_samples = uniform_samples.map(custom_cdfx) # Mapping uniform_samples to custom cdf samples generated from it
        cdfx_samples.rename(columns={'Uniform Samples': 'CDFX Samples'}, inplace=True) # Creating two columns
        cdfx_samples.to_csv("CDFX.csv") # Writing to CSV
        count = cdfx_samples['CDFX Samples'].value_counts().get(2.0, 0) # Counting number of occurences of two using a pandas function
        print(count)

        # Plotting histogram 
        plt.hist(cdfx_samples, bins=int(np.sqrt(N)))
        plt.title("CDFX Samples from Uniform_Samples")
        plt.xlabel("Frequency")
        plt.ylabel("CDFX Samples")
        plt.show()


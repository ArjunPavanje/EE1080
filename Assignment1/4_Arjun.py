import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducability
random.seed(42)

# Mode0: Angle made by chord w.r.t a tangent at one fixed end is equally likely
def Mode0(N):
    theta = [random.uniform(0, np.pi) for _ in range(N)] # Generating N uniform samples between 0, pi 
    length = 2*np.sin(theta) # Calculating length of chord from Angle 
    
    # Counting number of chords whose length is greater than root(3)
    count = 0 
    for i in length:
        if i >= np.sqrt(3):
            count += 1

    print(count/N) # Printing fraction 

    # Plotting Histogram 
    plt.hist(length, bins = int(np.sqrt(N))) 
    plt.title(f'Mode0: Angle made by chord w.r.t a tangent at one fixed end is equally likely')      
    plt.xlabel('Chord Length')
    plt.ylabel('Frequency')
    plt.show()

# Mode1: Distance of the chord from center is equally likely  
def Mode1(N):
    U = [random.uniform(0, 1) for _ in range(N)] # Generating N uniform samples between 0, pi 
    Y = 2*np.sqrt(1-np.pow(U, 2)) # Calculating length of chord from distance from centre 
    
    # Counting number of chords whose length is greater than root(3)
    count = 0 
    for i in Y:
        if i >= np.sqrt(3):
            count += 1

    print(count/N) # Printing fraction 

    # Plotting Histogram 
    plt.hist(Y, bins = int(np.sqrt(N))) 
    plt.title(f'Mode1: Distance of the chord from center is equally likely')      
    plt.xlabel('Chord Length')
    plt.ylabel('Frequency')
    plt.show()
    
# Mode2: Center of the chord is equally likely within circle
def Mode2(N):
    U = [random.uniform(0, 1) for _ in range(N)] # Generating N uniform samples between 0, pi 
    R = np.sqrt(U)
    Z = 2*np.sqrt(1-np.pow(R, 2)) # Calculating length of chord from distance from centre 
    
    # Counting number of chords whose length is greater than root(3)
    count = 0 
    for i in Z:
        if i >= np.sqrt(3):
            count += 1

    print(count/N) # Printing fraction 

    # Plotting Histogram 
    plt.hist(Z, bins = int(np.sqrt(N))) 
    plt.title(f'Mode2: Center of the chord is equally likely within circle')      
    plt.xlabel('Chord Length')
    plt.ylabel('Frequency')
    plt.show()


# Error Handling
if len(sys.argv) != 3:
    sys.exit("Invalid Number of Parameters")

# Accepting sys args
mode, N = int(sys.argv[1]), int(sys.argv[2])

# Error Handling
if mode != 0 and mode != 1 and mode != 2:
    sys.exit("mode should be either 0, 1 or 2")
if N < 0:
    sys.exit("N should be a non negative integer")

if mode == 0:
    Mode0(N)
elif mode == 1:
    Mode1(N)
else:
    Mode2(N)

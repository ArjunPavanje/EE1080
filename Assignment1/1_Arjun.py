import random
import sys

# Setting a random seed for reproducability
random.seed(42)

# Fixing maximum number of coin tosses a game can run for (to avoid overflow issues in the worst case)
max_toss = int(1e12)

# Simulating a coin toss (Returns 1 if heads, 0 if tails)
def coin_toss(p):
    n = random.uniform(0, 1) # Generating a random number between 0, 1 
    if n<p:
        return 1 
    else:
        return 0

# Simulating St Petersberg's Paradox
def St_Petersburg(m):
    avg_payout = 0
    p = 0.5 # Assuming fair coin 
    for _ in range(m): # m games
        for i in range(1, max_toss+1):
            toss = coin_toss(p)
            if toss == 0: # Stop the game if tails is encountered
                avg_payout += 2**i # Payout for a game that lasts k tosses is 2^k
                break
    
    return avg_payout/m 

# Printing average Payout for m = 100, 10000, 1000000 
print(round(St_Petersburg(100), 3), round(St_Petersburg(10000), 3), round(St_Petersburg(1000000), 3))


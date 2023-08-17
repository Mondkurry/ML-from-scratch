import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import threading
import traceback

def loadingAnimation(func) -> None:
    def wrapper(*args, **kwargs):
        thr = threading.Thread(target=func, args=(), kwargs={})
        try:
            thr.start() # Run Function 
            while thr.is_alive(): # While Function is running
                print("\r|", end="")
                time.sleep(0.1)
                print("\r/", end="")
                time.sleep(0.1)
                print("\r-", end="")
                time.sleep(0.1)
                print("\r\\", end="")
                time.sleep(0.1)
                print("\r", end="")  
            thr.join() # Will wait till function is done and join it.
            print(func.__name__ + " is Done!")
        except:
            traceback.print_exc()
            print("Error: unable to start thread")
    return wrapper

def printPurple(string): 
    """Literally just prints text in purple.. why not?"""
    print(f'\033[35m{string}\033[0m')

# Function to plot a given function
def plot_function(func, func_name, x_range=(-5, 5)):
    '''
    Plots a given function.
    
    Parameters:
    func (function): The function to be plotted.
    func_name (str): The name of the function.
    x_range (tuple): The range of x values to be plotted.
    '''
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], 100)

    # Calculate corresponding y values using the provided function
    y = func(x)

    # Create the plot
    plt.plot(x, y, label=func_name)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('x')
    plt.ylabel(f'{func_name}(x)')
    plt.title(f'{func_name} Function')
    plt.legend()
    plt.grid()
    plt.show()
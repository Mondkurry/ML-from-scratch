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
    
def displayPower(base , exponent) -> str:
    """Displays a power in a nice format"""
    return f'{base}{create_superscript(str(exponent))}'

def create_superscript(text) -> str:
    superscript_mapping = {
        '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
        '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
        '8': '\u2078', '9': '\u2079', '+': '\u207A', '-': '\u207B',
        '=': '\u207C', '(': '\u207D', ')': '\u207E',
        'a': '\u1D43', 'b': '\u1D47', 'c': '\u1D9C', 'd': '\u1D48',
        'e': '\u1D49', 'f': '\u1DA0', 'g': '\u1D4D', 'h': '\u02B0',
        'i': '\u2071', 'j': '\u02B2', 'k': '\u1D4F', 'l': '\u02E1',
        'm': '\u1D50', 'n': '\u207F', 'o': '\u1D52', 'p': '\u1D56',
        'q': '\u1D57', 'r': '\u02B3', 's': '\u02E2', 't': '\u1D57',
        'u': '\u1D58', 'v': '\u1D5B', 'w': '\u02B7', 'x': '\u02E3',
        'y': '\u02B8', 'z': '\u1DBB',
    }
    
    superscript_text = ''.join(superscript_mapping.get(char, char) for char in text)
    return superscript_text


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
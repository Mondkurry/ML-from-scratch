import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
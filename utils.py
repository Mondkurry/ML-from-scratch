import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(title, xlabel, ylabel, x, y, legend=None, grid=True):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend if legend else [])
    plt.grid(grid if grid else False)
    plt.show()
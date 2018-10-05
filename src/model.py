import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: train test split? where?

# TODO: Import data -> input: filepath -> numpy array
def import_data(filepath):
    return np.loadtxt(filepath)

# NOTE: Feature selection: perform in notebook?

# TODO: Feature clean-up -> input: x array -> cleaned x array
# TODO: Create pipeline -> input: x array, y array -> pipeline
# TODO: Grid search and cross-validate: input pipeline; return best hyper-parameters and score?
    
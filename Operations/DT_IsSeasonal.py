import numpy as np

def DT_IsSeasonal(y):
    """
    A simple test of seasonality.
    """

    N = len(y)
    r = np.arange(N)

    # Fit a sinusoidal model
    smodel = lambda x, a1, b1, c1 : a1 * np.sin(b1*x+c1)
    

import numpy as np
from Operations.CO_AutoCorr import CO_AutoCorr

def CO_FirstMin(y, minWhat='mi-gaussian', extraParam=None, minNotMax=True):
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    minWhat : str, optional
        The type of correlation to minimize. Options are 'ac' for autocorrelation,
        or 'mi' for automutual information. By default, 'mi' specifies the
        'gaussian' method from the Information Dynamics Toolkit. Other options
        include 'mi-kernel', 'mi-kraskov1', 'mi-kraskov2' (from Information Dynamics Toolkit),
        or 'mi-hist' (histogram-based method). Default is 'mi'.
    extraParam : any, optional
        An additional parameter required for the specified `minWhat` method (e.g., for Kraskov).
    minNotMax : bool, optional
        If True, return the maximum instead of the minimum. Default is False.

    Returns
    -------
    int
        The time of the first minimum (or maximum if `minNotMax` is True).
    """

    N = len(y)

    # Define the autocorrelation function
    if minWhat in ['ac', 'corr']:
        # Autocorrelation implemented as CO_AutoCorr
        corrfn = lambda x : CO_AutoCorr(y, tau=x, method='Fourier')
    elif minWhat == 'mi-hist':
        # do stuff
    elif minWhat == 'mi-kraskov2':
        # do stuff
    elif minWhat == 'mi-kraskov1':
        # do stuff
    elif minWhat == 'mi-kernel':
        # do stuff
    elif minWhat in ['mi', 'mi-gaussian']:
        # do stuff
    else:
        raise ValueError(f"Unknown correlation type specified: {minWhat}")
    
    # search for a minimum
    

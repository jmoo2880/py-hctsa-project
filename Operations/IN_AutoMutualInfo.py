import numpy as np
import warnings
import jpype as jp
from scipy import stats

def IN_AutoMutualInfo(y, timeDelay=1, estMethod='gaussian'):
    """
    Time-series automutual information

    Parameters:
    -----------
    y : array_like
        Input time series (column vector)
    time_delay : int or list, optional
        Time lag for automutual information calculation (default is 1)
    est_method : str, optional
        The estimation method used to compute the mutual information:
        - 'gaussian'
        - 'kernel'
        - 'kraskov1'
        - 'kraskov2'
        (default is 'kernel')
    extra_param : any, optional
        Extra parameters for the estimation method (default is None)

    Returns:
    --------
    out : float or dict
        Automutual information value(s)
    """

    if isinstance(timeDelay, str) and timeDelay in ['ac', 'tau']:
        print("ADD CO_FIRSTCROSSING")
    
    y = np.asarray(y).flatten()
    N = len(y)
    minSamples = 5 # minimum 5 samples to compute mutual information (could make higher?)

    # Loop over time delays if a vector
    if isinstance(timeDelay, int):
        numTimeDelays = 1
    else:   
        numTimeDelays = len(timeDelay)
        
    amis = np.full(numTimeDelays, np.nan)

    if numTimeDelays > 1:
        timeDelay = np.sort(timeDelay)
    
    # Initialize miCalc object (needs to be reinitialized within the loop for kraskov)
    # if estMethod == 'gaussian':
    #     # temporary placement
    #     jarloc = "/Users/joshua/Documents/jidt/infodynamics.jar"
    #     jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)

    for k, delay in enumerate(timeDelay):
        # check enough samples to compute automutual info
        if delay > N - minSamples:
            # time sereis too short - keep the remaining values as NaNs
            break

        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        if estMethod == 'gaussian':
            r, _ = stats.pearsonr(y1, y2)
            amis[k] = -0.5*np.log(1 - r**2)
        else:
            print("implement m,i calc")
        
        if np.isnan(amis).any():
            print(f"Warning: Time series (N={N}) is too short for automutual information calculations up to lags of {max(timeDelay)}")
        if numTimeDelays == 1:
            return amis[0]
        else:
            return {f"ami{delay}": ami for delay, ami in zip(timeDelay, amis)}

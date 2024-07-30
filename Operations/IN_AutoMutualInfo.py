import numpy as np
import warnings

def IN_AutoMutualInfo(y, timeDelay=1, estMethod='kernel', extraParam=None):
    """
    Time-series automutual information.
    """

    N = len(y)
    doPlot = False # plot outputs to the screen
    minSamples = 5 # minimum 5 samples to compute mutual information (could make higher?)

    # ensure y is a column vector
    if np.size(y, 1) > np.size(y, 0):
        warnings.warn("Please input a column vector for y")
        y = y.T

    # Loop over time delays if a vector
    numTimeDelays = len(timeDelay)
    amis = np.full((numTimeDelays, 1), np.nan)

    if numTimeDelays > 1:
        timeDelay = np.sort(timeDelay)
    
    # Initialize miCalc object (needs to be reinitialized within the loop for kraskov)
    

from PeripheryFunctions.BF_SignChange import BF_SignChange
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.IN_AutoMutualInfo import IN_AutoMutualInfo
import numpy as np
from scipy import stats

def IN_AutoMutualInfoStats(y, maxTau=None, estMethod='', extraParam=None):
    """
    Statistics on automutual information function of a time series.

    Parameters:
    ----------
    y (array-like) : column vector of time series.
    estMethod (str) : input to IN_AutoMutualInfo
    extraParam (str, int, optional) : input to IN_AutoMutualInfo
    maxTau (int) : maximal time delay

    Returns:
    --------
    out (dict) : a dictionary containing statistics on the AMIs and their pattern across the range of specified time delays.
    """

    N = len(y) # length of the time series
    
    # maxTau: the maximum time delay to investigate
    if maxTau is None:
        maxTau = np.ceil(N/4)
    maxTau0 = maxTau

    # Don't go above N/2
    maxTau = min(maxTau, np.ceil(N/2))

    # Get the AMI data
    timeDelay = list(range(1, maxTau+1))
    print(timeDelay)
    ami = IN_AutoMutualInfo(y, timeDelay=list(range(1, maxTau+1)), estMethod=estMethod, extraParam=extraParam)
    ami = np.array(list(ami.values()))

    out = {} # create dict for storing results
    # Output the raw values
    for i in range(1, maxTau0+1):
        if i <= maxTau:
            out[f'ami{i}'] = ami[i-1]
        else:
            out[f'ami{i}'] = np.nan

    # Basic statistics
    lami = len(ami)
    out['mami'] = np.mean(ami)
    out['stdami'] = np.std(ami)

    # First minimum of mutual information across range
    dami = np.diff(ami)
    extremai = np.where((dami[:-1] * dami[1:]) < 0)[0]
    out['pextrema'] = len(extremai) / (lami - 1)
    out['fmmi'] = min(extremai) + 1 if len(extremai) > 0 else lami

    return out 

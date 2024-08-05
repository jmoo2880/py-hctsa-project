import numpy as np
from scipy import stats, signal
from Operations.CO_AutoCorr import CO_AutoCorr

def SC_FluctAnal(x, q = 2, wtf = 'rsrange', tauStep = 1, k = 1, lag = None, logInc = True):
    """
    """

    N = len(x)

    # Compute integrated sequence
    if lag is None or lag == 1:
        y = np.cumsum(y)
    else:
        y = np.cumsum(x[::lag])
    
    # Perform scaling over a range of tau, up to a fifth the time-series length
    if logInc:
        taur = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(N // 2), tauStep))).astype(int))
    else:
        taur = np.arange(5, int(np.floor(N/2)) + 1, tauStep)
    ntau = len(taur)

    print(y)

import numpy as np


def PN_sampenc(y, M = 1, r = None, justM = False):
    """
    Calculate Sample Entropy

    Parameters:
    y (array-like): Input time-series data
    M (int): Maximum template length (embedding dimension)
    r (float, optional): Matching tolerance level. If None, defaults to 0.1 * std(y)
    justM (bool, optional): If True, return e just for the given M, not for all m up to it

    Returns:
    tuple: (e, p, A, B)
        e: Sample entropy estimates for m=0,1,...,M-1
        p: Probabilities
        A: Number of matches for m=1,...,M
        B: Number of matches for m=1,...,M excluding last point
    """
    if r is None:
        r = 0.1 * np.std(y, ddof=1)
    
    N = len(y)
    lastrun = np.zeros(N)
    run = np.zeros(N)
    A = np.zeros(M)
    B = np.zeros(M)
    p = np.zeros(M)
    e = np.zeros(M)

    # get counting 
    for i in range(N): # go through each point in the time series, counting matches
        y1 = y[i]
        for jj in range(N):
            # compare to future index, j
            j = i + jj 
            # this future point, j, matches the time series value at i
            if np.abs(y[j]-y1) < r:
                run[jj] = lastrun[jj] + 1 # increase run count for this lag
                M1 = min(M, run[jj])
                for m in range(M1+1):
                    A[m] = A[m] + 1
                    if j < N:
                        B[m] = B[m] + 1
            else:
                run[jj] = 0
        for j in range(N-i+1):
            lastrun[j] = run[j]
    

    return lastrun, run, A, B

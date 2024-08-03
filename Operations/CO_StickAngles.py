import numpy as np
from scipy import stats, signal

def CO_StickAngles(y):
    """
    """

    # Split the time series into positive and negative parts
    ix = [np.where(y >= 0)[0], np.where(y < 0)[0]]
    n = [len(ix_) for ix_ in ix]

    # Compute the stick angles
    angles = [np.zeros(n_-1) for n_ in n]
    for j in range(2):
        for i in range(n[j]-1):
            angles[j][i] = (y[ix[j][i+1]] - y[ix[j][i]]) / (ix[j][i+1] - ix[j][i])
        angles[j] = np.arctan(angles[j])
    all_angles = np.concatenate(angles)

    # Initialize output dictionary
    out = {}
    return all_angles

    
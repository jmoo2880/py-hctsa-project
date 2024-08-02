import numpy as np 
from Operations.SB_CoarseGrain import SB_CoarseGrain

def FC_Surprise(y, whatPrior = 'dist', memory = 0.2, numGroups = 3, coarseGrainMethod = 'quantile', numIters = 500, randomSeed = None):
    """
    FC_Surprise   How surprised you would be of the next data point given recent memory.

    Coarse-grains the time series, turning it into a sequence of symbols of a
    given alphabet size, num_groups, and quantifies measures of surprise of a
    process with local memory of the past memory values of the symbolic string.

    We then consider a memory length, memory, of the time series, and
    use the data in the proceeding memory samples to inform our expectations of
    the following sample.

    The 'information gained', log(1/p), at each sample using expectations
    calculated from the previous memory samples, is estimated.

    Parameters:
    -----------
    y : array_like
        The input time series
    what_prior : str, optional
        The type of information to store in memory:
        'dist': the values of the time series in the previous memory samples,
        'T1': the one-point transition probabilities in the previous memory samples,
        'T2': the two-point transition probabilities in the previous memory samples.
    memory : int or float, optional
        The memory length (either number of samples, or a proportion of the
        time-series length, if between 0 and 1) (default: 0.2)
    num_groups : int, optional
        The number of groups to coarse-grain the time series into (default: 3)
    coarse_grain_method : str, optional
        The coarse-graining, or symbolization method:
        'quantile': an equiprobable alphabet by the value of each time-series datapoint,
        'updown': an equiprobable alphabet by the value of incremental changes in the time-series values,
        'embed2quadrants': 4-letter alphabet of the quadrant each data point resides in a two-dimensional embedding space.
    num_iters : int, optional
        The number of iterations to repeat the procedure for.
    random_seed : int or None, optional
        Seed for the random number generator

    Returns:
    --------
    dict
        Summaries of the series of information gains, including the
        minimum, maximum, mean, median, lower and upper quartiles, and
        standard deviation.
    """

    y = np.array(y)
    N = len(y) # time series length

    # specify memory as a proportion of the time-series length
    if isinstance(memory, float) and (0 < memory < 1):
        # if float, then must be a proportion of the time series length
        memory = np.ceil(memory * N)
    
    # Course Grain
    yth = SB_CoarseGrain(y, coarseGrainMethod, numGroups)
    # Select random samples to test
    np.random.seed(randomSeed) # control random seed (for reproducibility)
    rs = np.random.permutation(N - memory) + memory # Can't do beginning of time series, up to memory
    rs.sort() # Just use a random sample of numIters points to test
    rs = rs[:min(numIters, len(rs))]

    # Compute empirical probabilities from time series
    store = np.zeros(len(rs))
    for i, r in enumerate(rs):
        if whatPrior == 'dist':
            p = np.mean(yth[r-memory:r] == yth[r])
        elif whatPrior == 'T1':
            memory_data = yth[r-memory:r]
            in_mem = np.where(memory_data[:-1] == yth[r-1])[0]
            p = np.mean(memory_data[in_mem+1] == yth[r]) if len(in_mem) > 0 else 0
        elif whatPrior == 'T2':
            memory_data = yth[r-memory:r]
            in_mem1 = np.where(memory_data[1:-1] == yth[r-1])[0]
            in_mem2 = np.where(memory_data[in_mem1] == yth[r-2])[0]
            p = np.mean(memory_data[in_mem2+2] == yth[r]) if len(in_mem2) > 0 else 0
        else:
            raise ValueError(f"Unknown method '{whatPrior}'")
        store[i] = p

    # Information gained from next observation is log(1/p) = -log(p)
    store[store == 0] = 1  # so that we set log(0) == 0
    store = -np.log(store)  # transform to surprises/information gains

    # Calculate statistics
    out = {
        'min': np.min(store[store > 0]) if np.any(store > 0) else np.nan,
        'max': np.max(store),
        'mean': np.mean(store),
        'sum': np.sum(store),
        'median': np.median(store),
        'lq': np.quantile(store, 0.25),
        'uq': np.quantile(store, 0.75),
        'std': np.std(store, ddof=1)
    }
    
    # t-statistic to information gain of 1
    out['tstat'] = np.abs((out['mean'] - 1) / (out['std'] / np.sqrt(numIters))) if out['std'] != 0 else np.nan

    return out

import numpy as np

def BF_PointOfCrossing(x, threshold):
    """
    Linearly interpolate to the point of crossing a threshold
    
    Parameters:
    x (array-like): a vector
    threshold (float): a threshold x crosses
    
    Returns:
    tuple: (firstCrossing, pointOfCrossing)
    firstCrossing (int): the first discrete value after which a crossing event has occurred
    pointOfCrossing (float): the (linearly) interpolated point of crossing
    """
    x = np.asarray(x)
    
    if x[0] > threshold:
        firstCrossing = np.where((x - threshold) < 0)[0][0]
    else:
        firstCrossing = np.where((x - threshold) > 0)[0][0]

    if firstCrossing.size == 0:
        # Never crosses
        N = len(x)
        firstCrossing = N
        pointOfCrossing = N
    else:
        # Continuous version---the point of crossing
        valueBeforeCrossing = x[firstCrossing - 1]
        valueAfterCrossing = x[firstCrossing]
        pointOfCrossing = firstCrossing - 1 + (threshold - valueBeforeCrossing) / (valueAfterCrossing - valueBeforeCrossing)

    return firstCrossing, pointOfCrossing

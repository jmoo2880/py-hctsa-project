import numpy as np

def BF_PointOfCrossing(x, threshold):
    """
    Linearly interpolate to the point of crossing a threshold

    Parameters:
    x (array-like): A vector of values
    threshold (float): A threshold x crosses

    Returns:
    tuple: (firstCrossing, pointOfCrossing)
    firstCrossing (int): The first discrete value after which a crossing event has occurred
    pointOfCrossing (float): The (linearly) interpolated point of crossing
    """
    x = np.array(x)  # Convert input to numpy array for consistency

    # Find index of x at which the first crossing event occurs
    if x[0] > threshold:
        firstCrossing = np.argmax(x - threshold < 0)
    else:
        firstCrossing = np.argmax(x - threshold > 0)

    if firstCrossing == 0 and not np.any(x - threshold < 0 if x[0] > threshold else x - threshold > 0):
        # Never crosses
        N = len(x)
        firstCrossing = N - 1  # Adjust to 0-based indexing
        pointOfCrossing = N - 1  # Adjust to 0-based indexing
    else:
        # Continuous version---the point of crossing
        valueBeforeCrossing = x[firstCrossing - 1]
        valueAfterCrossing = x[firstCrossing]
        pointOfCrossing = firstCrossing - 1 + (threshold - valueBeforeCrossing) / (valueAfterCrossing - valueBeforeCrossing)

    return firstCrossing, pointOfCrossing

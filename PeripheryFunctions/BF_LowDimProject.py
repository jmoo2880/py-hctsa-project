import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import zscore

def BF_LowDimProject(dataMatrix, whatAlgorithm='pca', numComponents=2):
    """
    BF_LowDimProject Compute low-dimensional components of a data matrix.
    
    Parameters
    ----------
    dataMatrix : numpy.ndarray
        Matrix of observations x variables.
    whatAlgorithm : str, optional
        The dimensionality reduction algorithm ('pca', 'tsne'), by default 'pca'.
    numComponents : int, optional
        The number of components to use, by default 2.
    Operations : pandas.DataFrame, optional
        DataFrame detailing the columns of dataMatrix (optional), by default None.
    
    Returns
    -------
    lowDimComponents : numpy.ndarray
        Low-dimensional components.
    componentLabels : list of str
        Text labels for each component.
    """
    numFeatures = np.size(dataMatrix, 1) # get the number of features as the number of columns in the dataMatrix

    # select a dim reduction method
    if whatAlgorithm in ['pca', 'PCA']:
        print(f"Calculating {numComponents}-dimensional principal components of the {np.size(dataMatrix, 0)} x {np.size(dataMatrix, 1)} data matrix...")
        # Use pca to compute the first two principal components: 
        # (project data into space of PC scores, Y)
        # normalize the data
        pca = PCA(n_components=numComponents)
        pca.fit(zscore(dataMatrix))
        lowDimComponents = pca.components_
        pcCoeff = pca.singular_values_
        percVar = pca.explained_variance_ratio_
        # display the features loading strongly onto the first two components


    elif whatAlgorithm in ['tsne', 'tSNE']:
        defaultNumPCs = 100
        numPCAComponents = min(np.size(dataMatrix, 1), defaultNumPCs)
        # set default rng for reproducibility
    else:
        raise ValueError(f'Unknown dimensionality-reduction algorithm: {whatAlgorithm}')

def LowDimDisplayTopLoadings(numTopLoadFeat, numPCs, pcCoeff, pcScore):
    # Display feature-loading-onto-PC info to screen
    

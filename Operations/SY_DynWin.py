import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.EN_SampEn import EN_SampEn
from scipy.stats import iqr, skew, kurtosis


def SY_DynWin(y, maxNumSegments):
    """
    """
    nsegr = np.arange(2, maxNumSegments+1, 1) # range of nseg to sweep across
    nmov = 1 # controls window overlap

    numFeatures = 11 # num of features
    fs = np.zeros((len(nsegr), numFeatures)) # standard deviation of feature values over windows
    taug = CO_FirstCrossing(y, 'ac', 0, 'discrete') # global tau

    for (i, nseg) in enumerate(nsegr):
        wlen = int(np.floor(len(y)/nseg)) # window length
        inc = int(np.floor(wlen/nmov)) # increment to move at each step
        # if increment is rounded to zero, prop it up
        if inc == 0:
            inc = 1
        
        numSteps = int(np.floor((len(y) - wlen)/inc) + 1)
        qs = np.zeros((numSteps, numFeatures))

        for j in range(numSteps):
            ySub = y[j*inc+1:j*inc+wlen]
            taul = CO_FirstCrossing(ySub, 'ac', 0, 'discrete')

            qs[j, 0] = np.mean(ySub)
            qs[j, 1] = np.std(ySub, ddof=1)
            qs[j, 2] = skew(ySub)
            qs[j, 3] = kurtosis(ySub)
            sampenOut = EN_SampEn(ySub, 2, 0.15)
            qs[j, 4] = sampenOut['quadSampEn1'] # SampEn_1_015
            qs[j, 5] = sampenOut['quadSampEn2'] # SampEn_2_015
            qs[j, 6] = CO_AutoCorr(ySub, 1, 'Fourier') # AC1
            qs[j, 7] = CO_AutoCorr(ySub, 2, 'Fourier') # AC2
            # (Sometimes taug or taul can be longer than ySub; then these will output NaNs:)
            qs[j, 8] = CO_AutoCorr(ySub, taug, 'Fourier') # AC_glob_tau
            qs[j, 9] = CO_AutoCorr(ySub, taul, 'Fourier') # AC_loc_tau
            qs[j, 10] = taul
        
        fs[i, :numFeatures] = np.std(qs, ddof=1)

    fs = np.std(fs, ddof=1)

    # Outputs
    out = {}
    out['stdmean'] = fs[1]
    out['stdstd'] = fs[2]
    out['stdskew'] = fs[3]
    out['stdkurt'] = fs[4]
    out['stdsampen1_015'] = fs[5]
    out['stdsampen2_015'] = fs[6]
    out['stdac1'] = fs[7]
    out['stdac2'] = fs[8]
    out['stdactaug'] = fs[9]
    out['stdactaul'] = fs[10]
    out['stdtaul'] = fs[11]

    return out 

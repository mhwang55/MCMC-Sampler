import numpy as np
import pandas as pd
import random as rand
import numba

# seed the random number generator
rand.seed()

# below function takes (or rejects) a Metropolis-Hastings step 
def mhStep(x, logPost, logPostCurrent, sigma, args=()):
    """
    Parameters
    ----------
    x : ndarray, shape (nVars,)
        present location of walker in parameter space.
    logPost : function
        function to compute log posterior. Has call signature
        `logPost(x, *args)`.
    logPostCurrent : float
        current value of log posterior.
    sigma : ndarray, shape (nVars, )
        standard deviations for proposal distribution.
    args : tuple
        other arguments passed to `logPost()` function.

    Returns
    -------
    xOut : ndarray, shape (nVars,)
        position of walker after Metropolis-Hastings
        step. If no step taken, returns inputted `x`.
    logPostUpdated : float
        log posterior after step.
    accepted : bool
        true if proposal step was taken, false otherwise.
    """
    y = np.empty(len(x))
    
    xOut = x
    
    # take random samples from a normal distribution for each element in x
    for i in range(len(x)):
        y[i] = np.random.normal(x[i], sigma)
        
    logPostNew = logPost(y, *args)
    
    # metropolis ratio (logPostNew and logPostCurrent are log probabilities)
    r = np.exp(logPostNew - logPostCurrent)
    
    accepted = np.random.rand() < r
    
    if accepted:
        xOut = y
        logPostCurrent = logPostNew
        
    return xOut, logPostCurrent, accepted

# below function that calls above step function over and over again to do the
# sampling
def mhSample(logPost, x0, sigma, args=(), nBurn=1000, nSteps=1000,
              varNames=None):
    """
    Parameters
    ----------
    logPost : function
        function to compute the log posterior. Has call
        signature `log_post(x, *args)`.
    x0 : ndarray, shape (nVars,)
        starting location of walker in parameter space.
    sigma : ndarray, shape (nVars, )
        standard deviations for proposal distribution.
    args : tuple
        additional arguments passed to `logPost()` function.
    nBurn : int, default 1000
        num of burn-in steps.
    nSteps : int, default 1000
        num of steps to take after burn-in.
    varNames : list, length nVars
        list of names of vars. If None, then var names are sequential integers.
    
    Returns
    -------
    output : DataFrame
        first `nVars` columns contain samples.  Additionally, column 'lnprob'
        has log posterior value at each sample.
    """
    
    postVal = logPost(x0, *args)
    
    samples = np.empty((nSteps, len(x0)))
    
    # throw out burn in samples
    for i in range(nBurn):
        x0, postVal, accept = mhStep(x0, logPost, postVal, sigma, args)
    
    # save our actual random walk steps
    for i in range(nSteps):
        samples[i], postVal, accept = mhStep(x0, logPost, postVal, sigma, args)
        x0 = samples[i]
    
    output = pd.DataFrame(samples, columns = varNames)
    
    return output

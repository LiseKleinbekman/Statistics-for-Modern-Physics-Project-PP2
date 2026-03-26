import numpy as np
import sys
from pathlib import Path
from scipy.special import gammaln


#Load the data from the data_load_funtion.py file 
# Add parent directory to path so we can import from data folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_load_function import load_data #import the function from the data folder

m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data() #call the function and unpack all the returned values
#print(m_center, counts, uncertainty, bin_width, m_lo, m_hi) #print to check that all the values are being loaded properly

# define log likelihood function for the Poisson distribution
def log_likelihood(model, params, m_center, counts, **kwargs):
    """
    Compute the Poisson log-likelihood for binned event counts.

    The Poisson likelihood is given by:
    L(mu) = (mu^n * e^(-mu)) / n!
    where n is the observed count (defined as counts) and mu is the expected count (model prediction). 
    
    Take the log of the likelihood to get the log-likelihood, because it is easier to work with the log:
    log L(mu) = n * log(mu) - mu - log(n!)

    Parameters
    ----------
    model : callable
        Function with signature:
            model(m_center, params, **kwargs) -> mu
        Must return expected counts per bin (same shape as counts).

    params : array-like
        These are the model parameters.

    m_center : array-like
        centers of the bins (independent variable).

    counts : array-like
        Observed counts per bin from the data.

    **kwargs :
        Additional keyword arguments passed to the model
        (e.g. bin_width) if needed.

    Returns
    -------
    float
        Total log-likelihood.
    """

    # Model prediction
    mu = model(m_center, params, **kwargs)

    # Ensure valid values (avoid log(0) and mu<0)
    mu = np.clip(mu, 1e-10, None)

    # Poisson log-likelihood
    return np.sum(counts * np.log(mu) - mu - gammaln(counts + 1))

#define a negative log-likelihood function, which is useful to minimize when fitting the model to the data. This allows us to minimize the negative log-likelihood and find the best fit parameters that maximaize the original log-likelihood.
def neg_log_likelihood(model, params, m_center, counts, **kwargs):
    """
    Negative log-likelihood (useful for minimizers).  
    """
    return -log_likelihood(model, params, m_center, counts, **kwargs)



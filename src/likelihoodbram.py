import numpy as np
from scipy.special import gammaln

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
            model(m_center, bin_width, *params, **kwargs) -> mu
        Must return expected counts per bin (same shape as counts).

    params : array-like
        These are the model parameters.

    m_center : array-like
        centers of the bins (independent variable).

    counts : array-like
        Observed counts per bin from the data.

    **kwargs :
        Additional keyword arguments passed to the model
        (e.g. `bin_width` and `sqrts`).

    Returns
    -------
    float
        Total log-likelihood.
    """

    bin_width = kwargs.get("bin_width")
    if bin_width is None:
        raise ValueError("bin_width must be provided as a keyword argument.")

    extra_kwargs = {k: v for k, v in kwargs.items() if k != "bin_width"}

    # Model prediction
    mu = model(m_center, bin_width, *params, **extra_kwargs)

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



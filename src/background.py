import numpy as np
from scipy.optimize import minimize
from likelihood import neg_log_likelihood
from data.data_load_function import load_data


# Load data using the function from data_load_function.py
m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data()

def x_variables(m, sqrts=13.0):
    """
    Compute the dimensionless variable x = m / sqrt(s).
    Parameters:
    m: array-like, mass values
    sqrts: float, center-of-mass energy in TeV (default 13.0)
    Returns:
    x: array-like, dimensionless variable
    """
    return m / sqrts

# Define the background model
def background(m, p0, p1, p2, p3, sqrts = 13.0):
    """
    Background model function
    Parameters:
    m: mass values
    p0, p1, p2, p3: model parameters, to be fitted
    sqrts: center-of-mass energy (13 TeV for this dataset)
    """
    x_variable = m / sqrts
    valid = x_variable > 0 and x_variable < 1
    # Guard against x >= 1 (unphysical) or x <= 0 
    x = x_variable[valid]
    f = p0 * (1 - x)**p1 * x**(p2 + p3 * np.log(x))
    return f

# Define a function to compute the predicted background counts in each bin
def model(m, bin_width, p1, p2, p3, p4, sqrts=13.0):
    """
    Predicted event counts in each bin using background_pdf * bin width.
    """
    shape = background(m, p1, p2, p3, p4, sqrts)
    return shape * bin_width

# Fit the background model to the data by minimizing the negative log-likelihood
def fit_background(mjj_centers, bin_widths, counts,
                   sqrts=13.0, p0=None, verbose=True):
    """
    Fit the background-only model to the observed counts using
    scipy.optimize.minimize (Nelder-Mead + L-BFGS-B refinement).
 
    Parameters
    ----------
    mjj_centers  : array, bin-centre masses in TeV
    bin_widths   : array, bin widths in TeV
    counts       : array, observed event counts
    sqrts        : float, sqrt(s) in TeV
    p0           : initial guess [p1, p2, p3, p4]
                   If None, sensible defaults are used.
    verbose      : print fit summary
 
    Returns
    -------
    result  : scipy OptimizeResult
    p_best  : best-fit parameter array
    mu_best : best-fit predicted counts array
    """
    #convert counts to a numpy array
    counts = np.asarray(counts, dtype=float)
 
    # --- initial parameter guess ---
    if p0 is None:
        # p1: rough normalisation (area under spectrum / typical shape value)
        x0 = x_variables(mjj_centers, sqrts = 13.0)
        p1_init = counts.sum() * np.median(bin_widths)  # order-of-magnitude
        p0 = [p1_init, 14.0, -5.0, -0.5] #initial guess for the parameters, based on typical values for this type of fit
 
    # bounds: p1 > 0, p2 > 0, p3 < 0 (falling spectrum), p4 unconstrained
    bounds = [(1e-6, None), (0, 50), (-20, 0), (-10, 10)]
 
    # define the objective function for minimization (negative log-likelihood)
    def objective(params):
        return neg_log_likelihood(params, mjj_centers, bin_widths, counts, sqrts)
 
    # two-stage: Nelder-Mead to explore, then L-BFGS-B to refine
    res1 = minimize(objective, p0, method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})
    result = minimize(objective, res1.x, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 10000, 'ftol': 1e-12})
 
    # Extract best-fit parameters
    p_best = result.x
    p1, p2, p3, p4 = p_best
 
    # Compute best-fit predicted counts
    mu_best = model(mjj_centers, bin_widths, p1, p2, p3, p4, sqrts)
 
    # Print fit summary 
    if verbose:
        print("=" * 55)
        print("Background-only fit (4-parameter)")
        print("=" * 55)
        print(f"  p1 (norm)  = {p_best[0]:.4e}")
        print(f"  p2         = {p_best[1]:.4f}")
        print(f"  p3         = {p_best[2]:.4f}")
        print(f"  p4         = {p_best[3]:.4f}")
        print(f"  -2 ln L    = {2 * result.fun:.2f}")
        print(f"  Converged  = {result.success}")
        print()
 
    return result, p_best, mu_best
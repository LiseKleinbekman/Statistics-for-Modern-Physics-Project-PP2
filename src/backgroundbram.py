import numpy as np
from numpy.testing import verbose
from scipy.optimize import minimize
from likelihoodbram import neg_log_likelihood
from pathlib import Path
import sys


#Load the data from the data_load_funtion.py file 
# Add parent directory to path so we can import from data folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_load_function import load_data #import the function from the data folder

m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data() #call the function and unpack all the returned values
#print(m_center, counts, uncertainty, bin_width, m_lo, m_hi) #print to check that all the values are being loaded properly



def x_variables(m, sqrts=13000.0):
    """
    Compute the dimensionless variable x = m / sqrt(s).
    Parameters:
    m: array-like, mass values in GeV
    sqrts: float, center-of-mass energy in GeV (default 13000.0 for 13 TeV)
    Returns:
    x: array-like, dimensionless variable
    """
    return m / sqrts

# Define the background model
def background(m, p0, p1, p2, p3, sqrts=13000.0):
    """
    Background model function.
    Parameters:
    m: mass values in GeV
    p0, p1, p2, p3: model parameters, to be fitted
    sqrts: center-of-mass energy in GeV (13000.0 for this dataset)
    """
    m = np.asarray(m, dtype=float)
    x_variable = m / sqrts
    valid = (x_variable > 0) & (x_variable < 1)

    # Guard against x >= 1 (unphysical) or x <= 0
    f = np.zeros_like(m, dtype=float)
    x = x_variable[valid]
    f[valid] = p0 * (1 - x)**p1 * x**(p2 + p3 * np.log(x))
    return f

# Define a function to compute the predicted background counts in each bin
def model(m, bin_width, p1, p2, p3, p4, sqrts=13000.0):
    """
    Predicted event counts in each bin using background_pdf * bin width.
    """
    shape = background(m, p1, p2, p3, p4, sqrts)
    return shape * bin_width


def numerical_hessian(func, params, epsilon=1e-5):
    """
    Compute numerical Hessian using central finite differences
    with adaptive step sizes per parameter.
    """
    params = np.array(params, dtype=float)
    n = len(params)
    hessian = np.zeros((n, n))

    # Adaptive step sizes
    eps = epsilon * np.maximum(1.0, np.abs(params))

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)

            ei[i] = eps[i]
            ej[j] = eps[j]

            f_pp = func(params + ei + ej)
            f_pm = func(params + ei - ej)
            f_mp = func(params - ei + ej)
            f_mm = func(params - ei - ej)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps[i] * eps[j])

    return hessian



# Fit the background model to the data by minimizing the negative log-likelihood
def fit_background(m_center, bin_width, counts,
                   sqrts=13000.0, p0=None, verbose=True):
    """
    Fit the background-only model to the observed counts using
    scipy.optimize.minimize (Nelder-Mead + L-BFGS-B refinement).
 
    Parameters
    ----------
    mjj_centers  : array, bin-centre masses in GeV
    bin_widths   : array, bin widths in GeV
    counts       : array, observed event counts
    sqrts        : float, sqrt(s) in GeV
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
        x0 = x_variables(m_center, sqrts = 13.0)
        p1_init = counts.sum() * np.median(bin_width)  # order-of-magnitude
        p0 = [p1_init, 14.0, -5.0, -0.5] #initial guess for the parameters, based on typical values for this type of fit
 
    # bounds: p1 > 0, p2 > 0, p3 < 0 (falling spectrum), p4 unconstrained
    bounds = [(1e-6, None), (0, 50), (-20, 0), (-10, 10)]
 
    # define the objective function for minimization (negative log-likelihood)
    def background_model_for_fit(mass_centers, bin_width, p1, p2, p3, p4, sqrts):
        return model(mass_centers, bin_width, p1, p2, p3, p4, sqrts)

    def objective(params):
        return neg_log_likelihood(
            background_model_for_fit,
            params,
            m_center,
            counts,
            bin_width=bin_width,
            sqrts=sqrts,
        )
 
    # two-stage: Nelder-Mead to explore, then L-BFGS-B to refine
    res1 = minimize(objective, p0, method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})
    result = minimize(objective, res1.x, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 10000, 'ftol': 1e-12})
 
    # Extract best-fit parameters
    p_best = result.x
    p1, p2, p3, p4 = p_best
 
    # Compute best-fit predicted counts
    mu_best = model(m_center, bin_width, p1, p2, p3, p4, sqrts)


    
    # Numerical Covariance Matrix and Parameter Uncertainties
    cov_matrix = None
    param_errors = np.full(len(p_best), np.nan)
    corr_matrix = None

    try:
        # Compute Hessian at best-fit point
        hessian = numerical_hessian(objective, p_best)

        # Invert Hessian → covariance matrix
        cov_matrix = np.linalg.inv(hessian)

        # Extract uncertainties
        diag = np.diag(cov_matrix)
        if np.all(diag > 0):
            param_errors = np.sqrt(diag)
        else:
            print("Warning: Covariance matrix has non-positive diagonal entries.")

        # Correlation matrix
        if np.all(param_errors > 0):
            corr_matrix = cov_matrix / np.outer(param_errors, param_errors)

    except np.linalg.LinAlgError:
        print("Warning: Hessian is singular and cannot be inverted.")
    except Exception as e:
        print(f"Warning: Numerical Hessian failed: {e}")


 
    # Print fit summary 
    if verbose:
        print("=" * 55)
        print("Background-only fit (4-parameter)")
        print("=" * 55)
        print(f"  p1 (norm)  = {p_best[0]:.4e} ± {param_errors[0]:.4e}")
        print(f"  p2         = {p_best[1]:.4f} ± {param_errors[1]:.4f}")
        print(f"  p3         = {p_best[2]:.4f} ± {param_errors[2]:.4f}")
        print(f"  p4         = {p_best[3]:.4f} ± {param_errors[3]:.4f}")
        print(f"  -2 ln L    = {2 * result.fun:.2f}")
        print(f"  Converged  = {result.success}")
        #print("Correlation matrix (if available):") #Print matrix to check 
        #print(result.hess_inv if result.hess_inv is not None else "Not available")
        print()

    return result, p_best, mu_best, cov_matrix, param_errors, corr_matrix

result, p_best, mu_best, cov_matrix, param_errors, corr_matrix = fit_background(m_center, bin_width, counts)

print("Best-fit parameters:")
for i, name in enumerate(["p1", "p2", "p3", "p4"]):
    print(f"{name} = {p_best[i]} ± {param_errors[i]}")
if corr_matrix is not None:
    print("Correlation matrix:")
    print(corr_matrix)
    print()

if cov_matrix is not None:
    print("Covariance matrix:")
    print(cov_matrix)
    print()

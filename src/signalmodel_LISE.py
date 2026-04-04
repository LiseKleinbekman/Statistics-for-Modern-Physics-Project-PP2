import numpy as np
from scipy.optimize import minimize
from backgroundbram import fit_background, model
from data.data_load_function import load_data
from likelihoodbram import neg_log_likelihood

# Create the signal pdf as a Gaussian function
def gaussian_signal_pdf(m, m0, sigma, A=1.0):
    """
    Un-normalised Gaussian signal shape evaluated at bin centres.
 
    Parameters
    ----------
    m   : array, bin-centre masses in TeV
    m0    : float, resonance mass in TeV
    sigma : float, Gaussian width in TeV  (> 0)
    A     : float, amplitude (default 1.0)
 
    Returns
    -------
    g : array, Gaussian shape  (peak = A)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    m = np.asarray(m, dtype=float)
    return A * np.exp(-0.5 * ((m - m0) / sigma)**2)

# Create the expected signal counts per bin by normalising the Gaussian 
def signal_counts_per_bin(m_center, bin_width, N_sig, m0, sigma, A=1.0):
    """
    Expected signal event counts per bin.

    The Gaussian is area-normalised to 1 (integral over m = 1),
    then scaled by N_sig.  Because we work with discrete bins:

        s_i = N_sig * G(m_i; m0, sigma) * Delta_m_i
              / sum_j [ G(m_j; m0, sigma) * Delta_m_j ]

    This ensures sum_i s_i = N_sig exactly (bin-normalisation).

    Parameters
    ----------
    m_center    : array, bin-centre masses in TeV
    bin_width   : array, bin widths in TeV
    N_sig       : float, total signal events (counts)
    m0          : float, resonance mass in TeV
    sigma       : float, Gaussian width in TeV
    A           : float, amplitude (default 1.0)

    Returns
    -------
    s : array, expected signal counts per bin
    """
    # Ensure inputs are numpy arrays for element-wise operations
    m_center = np.asarray(m_center, dtype=float)
    bin_width  = np.asarray(bin_width,  dtype=float)

    # Compute the un-normalised Gaussian shape at the bin centres
    g = gaussian_signal_pdf(m_center, m0, sigma, A)
    # Normalise the Gaussian shape to get expected counts per bin
    norm = np.sum(g * bin_width)

    # Guard against zero or negative normalisation (should not happen for a well-defined Gaussian)
    if norm <= 0:
        return np.zeros_like(m_center)

    return N_sig * g * bin_width / norm

# Set the standard deviation of the Gaussian signal shape as a fraction of the resonance mass
def default_sigma(m0, resolution_fraction=0.05):
    """
    Default signal width = detector mass resolution * m0.
 
    The ATLAS dijet mass resolution is roughly 5% of m (can be up to ~10%
    at low mass).  For a narrow resonance (intrinsic width << detector
    resolution) use resolution_fraction = 0.05.
    For a broader signal (e.g. Z' with 15% intrinsic width) increase this.
 
    Parameters
    ----------
    m0                  : float, resonance mass in TeV
    resolution_fraction : float, sigma/m0  (default 0.05)
 
    Returns
    -------
    sigma : float, Gaussian width in TeV
    """
    return resolution_fraction * m0

# Define model for combined signal + background counts in each bin
def predicted_sb_counts(m_center, bin_width, bkg_params, N_sig, m0, sigma,
                        n_bkg_params=4, sqrt_s_TeV=13000.0, A=1.0):
    """
    Total predicted counts for the signal + background model.
 
        mu_i^{S+B} = mu_i^{bkg} + s_i
 
    Parameters
    ----------
    m_center    : array, bin-centre masses in TeV
    bin_widths  : array, bin widths in TeV
    bkg_params  : array [p1, p2, p3] or [p1, p2, p3, p4]
    N_sig       : float, total signal events
    m0          : float, resonance mass in TeV
    sigma       : float, Gaussian width in TeV
    n_bkg_params: 3 or 4
    sqrt_s_TeV  : float
    A           : float, amplitude (default 1.0)
 
    Returns
    -------
    mu : array, total predicted counts per bin
    """
    # n_bkg_params is 4 for us, but 3 in scale down. Handle both cases for flexibility.
    if n_bkg_params == 4:
        p1, p2, p3, p4 = bkg_params
    else:
        p1, p2, p3 = bkg_params
        p4 = 0.0
 
    # Compute the predicted background counts in each bin
    mu_bkg = model(m_center, bin_width, p1, p2, p3, p4, sqrt_s_TeV)
    s      = signal_counts_per_bin(m_center, bin_width, N_sig, m0, sigma, A)
    return mu_bkg + s


def numerical_hessian(func, params, epsilon=1e-4):
    """
    More stable numerical Hessian with adaptive step sizes.
    """
    params = np.array(params, dtype=float)
    n = len(params)
    hessian = np.zeros((n, n))

    # Better scaling (IMPORTANT)
    eps = epsilon * (1.0 + np.abs(params))

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

def fit_signal_plus_background(m_center, bin_width, counts,
                                bkg_p_init, m0, sigma=None,
                                n_bkg_params=4, sqrt_s_TeV=13000.0,
                                fix_m0=True, fix_sigma=True,
                                verbose=False):
    """
    Fit the S+B model for a given (or free) resonance mass m0.
 
    In the "profile-likelihood scan" mode (fix_m0=True), m0 is held fixed
    at the supplied value and the function is called on a grid of m0 values.
    This is the "scale-down option" described in the project brief.
 
    Parameters
    ----------
    m_center     : array, bin centres in TeV
    bin_width    : array, bin widths in TeV
    counts       : array, observed counts
    bkg_p_init   : array, initial background parameters from background-only fit
    m0           : float, resonance mass hypothesis in TeV
    sigma        : float or None
                   If None, uses default_sigma(m0) = 0.05 * m0.
    n_bkg_params : 3 or 4
    sqrt_s_TeV   : float
    fix_m0       : bool, hold m0 fixed (True = grid scan, False = free fit)
    fix_sigma    : bool, hold sigma fixed at supplied value
    verbose      : bool
 
    Returns
    -------
    result     : scipy OptimizeResult
    p_best_s     : best-fit parameter array  [bkg_params..., N_sig, (m0), (sigma)]
    mu_best _s   : best-fit total predicted counts
    nll_best_s   : minimum NLL value
    """
    counts = np.asarray(counts, dtype=float)

    # Set default sigma if not provided (often 5% of m0)
    if sigma is None:
        sigma = default_sigma(m0)
 
    # initial guess: start from background-only fit
    p0 = list(bkg_p_init[:n_bkg_params]) + [0.001 * np.sum(counts)]

    # add m0 and sigma to the parameter list if they are not fixed 
    if not fix_m0:
        p0 += [m0]
    if not fix_sigma:
        p0 += [sigma]
 
    # bounds (esures background parameters are physical, N_sig >= 0, 
    # m0 within the mass range, sigma > 0)
    bounds = [(1e-6, None)]   # p1
    bounds += [(1, 40)]       # p2 (avoid hitting 50 wall)
    bounds += [(-15, -0.1)]   # p3 (FORCE negative, avoid 0!)
    bounds += [(-5, 5)]       # p4 (reduce freedom)
    bounds += [(0, 1e5)]      # N_sig (prevent blow-up) 

    if not fix_m0:
        bounds += [(m_center.min(), m_center.max())]
    if not fix_sigma:
        bounds += [(1e-4, 2.0)]

    # define the objective function for minimization (negative log-likelihood)
    def sb_model_for_fit(mass_centers, bin_width, *opt_params,
                         n_bkg_params, sqrt_s_TeV,
                         fixed_m0=None, fixed_sigma=None):
        opt_params = np.asarray(opt_params, dtype=float)
        idx = 0
        bkg_params = opt_params[idx:idx + n_bkg_params]; idx += n_bkg_params
        N_sig = opt_params[idx]; idx += 1

        if fixed_m0 is None:
            m0_value = opt_params[idx]
            idx += 1
        else:
            m0_value = fixed_m0

        if fixed_sigma is None:
            sigma_value = opt_params[idx]
        else:
            sigma_value = fixed_sigma

        return predicted_sb_counts(
            mass_centers,
            bin_width,
            bkg_params,
            N_sig,
            m0_value,
            sigma_value,
            n_bkg_params=n_bkg_params,
            sqrt_s_TeV=sqrt_s_TeV,
            A=1.0,
        )

    def objective(params):
        return neg_log_likelihood(
            sb_model_for_fit,
            params,
            m_center,
            counts,
            bin_width=bin_width,
            n_bkg_params=n_bkg_params,
            sqrt_s_TeV=sqrt_s_TeV,
            fixed_m0=m0 if fix_m0 else None,
            fixed_sigma=sigma if fix_sigma else None,
        )
 
    # two-stage: Nelder-Mead to explore, then L-BFGS-B to refine (same as background fit)
    res1 = minimize(objective, p0, method='Nelder-Mead',
                    options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})
    result = minimize(objective, res1.x, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 10000, 'ftol': 1e-12})
 
    # Extract best-fit parameters 
    p_best_s = result.x
 
    # reconstruct mu_best from the best-fit parameters
    idx = 0
    bkg_best = p_best_s[idx:idx + n_bkg_params]; idx += n_bkg_params
    N_sig_best = p_best_s[idx]; idx += 1
    # m0_best and sigma_best depend on whether they were fixed or free 
    m0_best    = m0    if fix_m0    else p_best_s[idx]
    sigma_best = sigma if fix_sigma else p_best_s[idx + (0 if fix_m0 else 1)]
    # extract m0 if free and update index if needed
    if not fix_m0:
        m0_best = p_best_s[idx]; idx += 1
    # extract sigma if free and update index if needed
    if not fix_sigma:
        sigma_best = p_best_s[idx]
 
    # extract mu_best using the best-fit parameters 
    mu_best_s = predicted_sb_counts(m_center, bin_width,
                                   bkg_best, N_sig_best, m0_best, sigma_best,
                                   n_bkg_params, sqrt_s_TeV, A=1.0)
 


    # Numerical Hessian covariance (S+B)

    cov_matrix = None
    param_errors = np.full(len(p_best_s), np.nan)
    corr_matrix = None

    try:
        hessian = numerical_hessian(objective, p_best_s)

        # --- Regularisation (KEY FIX) ---
        reg = 1e-6 * np.eye(len(p_best_s))
        hessian_reg = hessian + reg

        # Try normal inverse first
        try:
            cov_matrix = np.linalg.inv(hessian_reg)
        except np.linalg.LinAlgError:
            print("Using pseudo-inverse for covariance...")
            cov_matrix = np.linalg.pinv(hessian_reg)

        diag = np.diag(cov_matrix)

        if np.all(diag > 0):
            param_errors = np.sqrt(diag)
        else:
            print("Warning: Non-positive diagonal in covariance.")
            param_errors = np.sqrt(np.abs(diag))

        if np.all(param_errors > 0):
            corr_matrix = cov_matrix / np.outer(param_errors, param_errors)

    except Exception as e:
        print(f"Warning: Numerical Hessian failed: {e}")

    # print fit summary (This will be printed everytime the function is called. Printing is done outside the function right now, but can be moved inside using the following code.)

    if verbose:
        print(f"  S+B fit  m0={m0_best:.3f} TeV  sigma={sigma_best:.4f} TeV")

        idx = 0
        for i in range(n_bkg_params):
            print(f"  p{i+1} = {p_best_s[idx]:.4e} ± {param_errors[idx]:.4e}")
            idx += 1

        print(f"  N_sig = {p_best_s[idx]:.4e} ± {param_errors[idx]:.4e}")
        idx += 1

        if not fix_m0:
            print(f"  m0    = {p_best_s[idx]:.4f} ± {param_errors[idx]:.4f}")
            idx += 1
        else:
            print(f"  m0 (fixed) = {m0_best:.4f}")

        if not fix_sigma:
            print(f"  sigma = {p_best_s[idx]:.4f} ± {param_errors[idx]:.4f}")
        else:
            print(f"  sigma (fixed) = {sigma_best:.4f}")

        print(f"  -2 ln L = {2 * result.fun:.2f}")
        print()

        if corr_matrix is not None:
            print("Correlation matrix:")
            print(corr_matrix)
            print()

        if cov_matrix is not None:
            print("Covariance matrix:")
            print(cov_matrix)
            print()

    nll_best = result.fun
    return result, p_best_s, mu_best_s, m0_best, nll_best, cov_matrix, param_errors, corr_matrix

#define all parameters
if __name__ == "__main__":
    m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data() 
    p_best_b = fit_background(m_center, bin_width, counts[1], p0=[10, 100, 16, 0.5], verbose=False)[1] #get the best fit background parameters from the background fit function. This will be used as an input for the signal+background fit function.
    result, p_best_s, mu_best_s, m0_best, nll_best, cov_matrix, param_errors, corr_matrix = fit_signal_plus_background(
        m_center, bin_width, counts,
        bkg_p_init=p_best_b,   
        m0=3.0,                         
        fix_m0=True,                    
        fix_sigma=True, verbose = False
    ) 

    print("=" * 55)
    print("Signal + Background fit results")
    print("=" * 55)

    idx = 0

    # Background parameters
    for i in range(4):  # or n_bkg_params if variable
        print(f"p{i+1} = {p_best_s[idx]:.4e} ± {param_errors[idx]:.4e}")
        idx += 1

    # Signal yield
    print(f"N_sig = {p_best_s[idx]:.4e} ± {param_errors[idx]:.4e}")
    idx += 1

    # m0
    if len(p_best_s) > idx:
        print(f"m0 = {p_best_s[idx]:.4f} ± {param_errors[idx]:.4f}")
        idx += 1
    else:
        print(f"m0 (fixed) = {m0_best:.4f}")

    # sigma
    if len(p_best_s) > idx:
        print(f"sigma = {p_best_s[idx]:.4f} ± {param_errors[idx]:.4f}")
    else:
        print("sigma (fixed)")

    print(f"-2 ln L = {2 * nll_best:.2f}")
    print()

    if corr_matrix is not None:
        print("Correlation matrix:")
        print(corr_matrix)
        print()

    if cov_matrix is not None:
        print("Covariance matrix:")
        print(cov_matrix)
        print()

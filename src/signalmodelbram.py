import numpy as np
from scipy.optimize import minimize
from backgroundbram import model
from likelihoodbram import neg_log_likelihood

# Create the signal pdf as a Gaussian function
def gaussian_signal_pdf(mjj, m0, sigma, A=1.0):
    """
    Un-normalised Gaussian signal shape evaluated at bin centres.
 
    Parameters
    ----------
    mjj   : array, bin-centre masses in TeV
    m0    : float, resonance mass in TeV
    sigma : float, Gaussian width in TeV  (> 0)
    A     : float, amplitude (default 1.0)
 
    Returns
    -------
    g : array, Gaussian shape  (peak = A)
    """
    # Ensure inputs are numpy arrays for element-wise operations
    mjj = np.asarray(mjj, dtype=float)
    return A * np.exp(-0.5 * ((mjj - m0) / sigma)**2)

# Create the expected signal counts per bin by normalising the Gaussian 
def signal_counts_per_bin(m_center, bin_width, N_sig, m0, sigma, A=1.0):
    """
    Expected signal event counts per bin.

    The Gaussian is area-normalised to 1 (integral over mjj = 1),
    then scaled by N_sig.  Because we work with discrete bins:

        s_i = N_sig * G(mjj_i; m0, sigma) * Delta_mjj_i
              / sum_j [ G(mjj_j; m0, sigma) * Delta_mjj_j ]

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
 
    The ATLAS dijet mass resolution is roughly 5% of mjj (can be up to ~10%
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
def predicted_sb_counts(mjj_centers, bin_widths, bkg_params, N_sig, m0, sigma,
                        n_bkg_params=4, sqrt_s_TeV=13000.0, A=1.0):
    """
    Total predicted counts for the signal + background model.
 
        mu_i^{S+B} = mu_i^{bkg} + s_i
 
    Parameters
    ----------
    mjj_centers : array, bin-centre masses in TeV
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
    mu_bkg = model(mjj_centers, bin_widths, p1, p2, p3, p4, sqrt_s_TeV)
    s      = signal_counts_per_bin(mjj_centers, bin_widths, N_sig, m0, sigma, A)
    return mu_bkg + s

def fit_signal_plus_background(mjj_centers, bin_widths, counts,
                                bkg_p_init, m0, sigma=None,
                                n_bkg_params=4, sqrt_s_TeV=13000.0,
                                fix_m0=True, fix_sigma=True,
                                verbose=True):
    """
    Fit the S+B model for a given (or free) resonance mass m0.
 
    In the "profile-likelihood scan" mode (fix_m0=True), m0 is held fixed
    at the supplied value and the function is called on a grid of m0 values.
    This is the "scale-down option" described in the project brief.
 
    Parameters
    ----------
    mjj_centers  : array, bin centres in TeV
    bin_widths   : array, bin widths in TeV
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
    p_best     : best-fit parameter array  [bkg_params..., N_sig, (m0), (sigma)]
    mu_best    : best-fit total predicted counts
    nll_best   : minimum NLL value
    """
    counts = np.asarray(counts, dtype=float)

    # Set default sigma if not provided (often 5% of m0)
    if sigma is None:
        sigma = default_sigma(m0)
 
    # initial guess: start from background-only fit
    p0 = list(bkg_p_init[:n_bkg_params]) + [10.0]   # N_sig = 10 as start

    # add m0 and sigma to the parameter list if they are not fixed 
    if not fix_m0:
        p0 += [m0]
    if not fix_sigma:
        p0 += [sigma]
 
    # bounds (esures background parameters are physical, N_sig >= 0, 
    # m0 within the mass range, sigma > 0)
    bounds = [(1e-6, None)] * 1  # p1
    bounds += [(0, 50), (0, 20), (-10, 10)]       # p2, p3, p4
    bounds += [(0, None)]                          # N_sig >= 0

    if not fix_m0:
        bounds += [(mjj_centers.min(), mjj_centers.max())]
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
            mjj_centers,
            counts,
            bin_width=bin_widths,
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
    p_best = result.x
 
    # reconstruct mu_best from the best-fit parameters
    idx = 0
    bkg_best = p_best[idx:idx + n_bkg_params]; idx += n_bkg_params
    N_sig_best = p_best[idx]; idx += 1
    # m0_best and sigma_best depend on whether they were fixed or free 
    m0_best    = m0    if fix_m0    else p_best[idx]
    sigma_best = sigma if fix_sigma else p_best[idx + (0 if fix_m0 else 1)]
    # extract m0 if free and update index if needed
    if not fix_m0:
        m0_best = p_best[idx]; idx += 1
    # extract sigma if free and update index if needed
    if not fix_sigma:
        sigma_best = p_best[idx]
 
    # extract mu_best using the best-fit parameters 
    mu_best = predicted_sb_counts(mjj_centers, bin_widths,
                                   bkg_best, N_sig_best, m0_best, sigma_best,
                                   n_bkg_params, sqrt_s_TeV, A=1.0)
 
    # print fit summary
    if verbose:
        print(f"  S+B fit  m0={m0_best:.3f} TeV  sigma={sigma_best:.4f} TeV")
        print(f"  N_sig = {N_sig_best:.1f}")
        print(f"  -2 ln L = {2 * result.fun:.2f}")
        print()
 
    return result, p_best, mu_best, m0_best, result.fun

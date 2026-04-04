from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_load_function import load_data
from backgroundbram import fit_background
from signalmodel_LISE import (
    fit_signal_plus_background,
    signal_counts_per_bin,
    default_sigma,
)



# Load data using the function from data_load_function.py
m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data()

#initial guess for the parameters, can be adjusted based on the data
Initial_guess = [10, 100, 16, 0.5] 

 
bkg_result, bkg_p_best, bkg_mu_best, cov_matrix, param_errors, corr_matrix = fit_background(m_center, bin_width, 
    counts, p0=Initial_guess, verbose=False)

#Define a function which calculates the background curve and resonance peak at certain values
def resonance_peak(m0, bkg_p_best, bkg_mu_best):
    """Calculate the background curve and resonance peak for a given m0 value.
    
    parameters:
    m0 : float which is the resonance mass hypothesis in GeV.
    bkg_p_best : the best fit parameters for the background model.
    bkg_mu_best : the best fit background curve.

    Returns:
    bkg_mu_best : the best fit background curve.
    resonance_visual_mu : the best fit background curve plus the resonance peak at m0.
    """

    signal_m0_2 = m0
    signal_sigma = default_sigma(signal_m0_2)

    # Fit the background and signal+background models to the data

    sb_result, sb_p_best, sb_mu_best, sb_m0_best, sb_nll_best, cov_matrix, param_errors, corr_matrix = fit_signal_plus_background(m_center,
        bin_width, counts, bkg_p_init=bkg_p_best, m0=signal_m0_2, sigma=signal_sigma,
        n_bkg_params=4, sqrt_s_TeV=13000.0, fix_m0=True, fix_sigma=True, verbose=False)

    # Build an illustrative resonance bump that peaks about one decade above
    signal_plot_sigma = 120.0
    unit_signal_shape = signal_counts_per_bin(m_center, bin_width, 1.0, signal_m0_2,
        signal_plot_sigma,)
    peak_idx = np.argmin(np.abs(m_center - signal_m0_2))
    target_signal_peak = bkg_mu_best[peak_idx]
    signal_plot_N = target_signal_peak / max(unit_signal_shape[peak_idx], 1e-12)
    signal_only_mu = signal_counts_per_bin(m_center, bin_width, signal_plot_N, signal_m0_2,
    signal_plot_sigma,)
    resonance_visual_mu = bkg_mu_best + signal_only_mu

    return bkg_mu_best, resonance_visual_mu


#now we make a function to find the m0 for which the resonance peak is maximum.
# def resonance_peak_maximum(m0_initial):
    """Iteratively find the m0 value that gives the maximum resonance peak. 
    Difference to the function above being that m0 is not fixed.
    
    parameters: 
    m0_initial : float which is the initial guess for the m0 value.
    
    Returns: 
    bkg_mu_best : the best fit background curve.
    resonance_visual_mu : the best fit background curve plus the resonance peak at the best fit m0 value.
    sb_m0_best : the best fit m0 value which gives the maximum resonance peak.
    """
#     signal_m0_2 = m0_initial
#     signal_sigma = default_sigma(signal_m0_2)
#     i = 0

#     while i < 100:
#         # Fit the background and signal+background models to the data
#         bkg_result, bkg_p_best, bkg_mu_best = fit_background(m_center, bin_width, 
#             counts, p0=Initial_guess, verbose=True)
#         sb_result, sb_p_best, sb_mu_best, sb_m0_best, sb_nll_best = fit_signal_plus_background(m_center,
#             bin_width, counts, bkg_p_init=bkg_p_best, m0=signal_m0_2, sigma=signal_sigma,
#             n_bkg_params=4, sqrt_s_TeV=13000.0, fix_m0=False, fix_sigma=True, verbose=True,)

#         # Build an illustrative resonance bump that peaks about one decade above
#         signal_plot_sigma = 120.0
#         unit_signal_shape = signal_counts_per_bin(m_center, bin_width, 1.0, signal_m0_2,
#             signal_plot_sigma,)
#         peak_idx = np.argmin(np.abs(m_center - signal_m0_2))
#         target_signal_peak = bkg_mu_best[peak_idx]
#         signal_plot_N = target_signal_peak / max(unit_signal_shape[peak_idx], 1e-12)
#         signal_only_mu = signal_counts_per_bin(m_center, bin_width, signal_plot_N, signal_m0_2,
#         signal_plot_sigma,)
#         resonance_visual_mu = bkg_mu_best + signal_only_mu
    
#         signal_m0_2 = sb_m0_best
#         signal_sigma = default_sigma(signal_m0_2)
#         i = i + 1
#         print(sb_m0_best)
#         print(i)

#     return bkg_mu_best, resonance_visual_mu, sb_m0_best


#Examples of m0 resonances at 2 and 3 TeV
background_curve, resonance_2000 = resonance_peak(2000, bkg_p_best, bkg_mu_best)
_, resonance_3000 = resonance_peak(3000, bkg_p_best, bkg_mu_best)


# Plot background fit with smaller data markers and keep the red line on top.
plt.errorbar(m_center, counts, yerr=uncertainty, fmt='o', color='black', markersize=3,
    elinewidth=1, capsize=2, alpha=0.8, label='Data', zorder=2,)
plt.plot(m_center, resonance_2000, color='green', linestyle='--', linewidth=2,
    label='Resonance peak at 2 TeV', zorder=1,)
plt.plot(m_center, resonance_3000, color='purple', linestyle='--', linewidth=2,
    label='Resonance peak at 3 TeV', zorder=1,)
plt.plot(m_center, background_curve, drawstyle='steps-mid',color='deepskyblue', linestyle='-', linewidth=2.5,
    label='Background Fit', zorder=1,)
# plt.plot(m_center, resonance_max, color='red', linestyle='-', linewidth=2.5, 
#          label = 'Maximum resonance peak at m0 = {:.1f} GeV'.format(m0_max), zorder=1,)
plt.xlabel('Mass [GeV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.title('Background + Signal Fit with Gaussian Bumps')
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'data_background.png', dpi=300)
plt.show()

#Gives the z-score of the data points. 
significance = (counts - background_curve) / np.sqrt(np.maximum(background_curve, 1e-10))

plt.figure()
plt.bar(m_center, significance, width=bin_width, alpha=0.6,
        edgecolor='black', label=r'$(N_{data} - N_{bkg}) / \sqrt{N_{bkg}}$')

plt.axhline(0.0, linestyle='--', linewidth=1)
plt.axhline(3.0, linestyle=':', linewidth=1, label=r'$\pm 3\sigma$ guide', color='red')
plt.axhline(-3.0, linestyle=':', linewidth=1, color='red')

plt.xlabel('Mass [GeV]')
plt.ylabel('Significance [$\sigma$]')
plt.title('Local Significance Relative to the Background Fit')
plt.legend()
plt.tight_layout()
plt.show()

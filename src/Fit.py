from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_load_function import load_data
from backgroundbram import fit_background
from signalmodelbram import (
    fit_signal_plus_background,
    signal_counts_per_bin,
    default_sigma,
)



# Load data using the function from data_load_function.py
m_center, counts, uncertainty, bin_width, m_lo, m_hi = load_data()

Initial_guess = [10, 100, 16, 0.5] #initial guess for the parameters, can be adjusted based on the data

def resonance_peak(m0):
    signal_m0_2 = m0
    signal_sigma = default_sigma(signal_m0_2)

    # Fit the background and signal+background models to the data
    bkg_result, bkg_p_best, bkg_mu_best = fit_background(m_center, bin_width, 
        counts, p0=Initial_guess, verbose=True)
    sb_result, sb_p_best, sb_mu_best, sb_nll_best = fit_signal_plus_background(m_center,
        bin_width, counts, bkg_p_init=bkg_p_best, m0=signal_m0_2, sigma=signal_sigma,
        n_bkg_params=4, sqrt_s_TeV=13000.0, fix_m0=True, fix_sigma=True, verbose=True,)

    # Build an illustrative resonance bump that peaks about one decade above
    # the fitted background around 2000 GeV on the log-scale plot.
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

background_curve, resonance_2000 = resonance_peak(2000)
_, resonance_3000 = resonance_peak(3000)

# Plot background fit with smaller data markers and keep the red line on top.
plt.errorbar(m_center, counts, yerr=uncertainty, fmt='o', color='black', markersize=3,
    elinewidth=1, capsize=2, alpha=0.8, label='Data', zorder=2,)
plt.plot(m_center, resonance_2000, color='green', linestyle='--', linewidth=2,
    label='Resonance peak at 2 TeV', zorder=1,)
plt.plot(m_center, resonance_3000, color='purple', linestyle='--', linewidth=2,
    label='Resonance peak at 3 TeV', zorder=1,)
plt.plot(m_center, background_curve, color='deepskyblue', linestyle='-', linewidth=2.5,
    label='Background Fit', zorder=1,)
plt.xlabel('Mass [GeV]')
plt.ylabel('Counts')
plt.yscale('log')
plt.title('Background + Signal Fit with Gaussian Bumps')
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'data_background.png', dpi=300)
plt.show()


significance = (counts - background_curve) / np.sqrt(uncertainty)

plt.figure()
plt.bar(m_center, significance, width=bin_width, alpha=0.6, color='steelblue',
    edgecolor='black', label='Data - Background Fit',)
plt.axhline(0.0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Mass [GeV]')
plt.ylabel('Residual Counts')
plt.title('Residuals of the Background Fit')
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'residuals_background.png', dpi=300)
plt.show()

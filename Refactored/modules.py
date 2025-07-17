"""
CV Eclipse Timing & O-C Analysis Modules

This module contains all the helper functions and core analysis routines
for the CV Eclipse Timing & O-C Pipeline.

Author: CV-Analyzer Team
Version: 5.0
Date: 2025-07-16
"""

from __future__ import annotations
import os
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks

from astropy.time import Time
from astropy.timeseries import LombScargle
import astropy.units as u

# Lightkurve (>=2.5)
import lightkurve as lk
from lightkurve.correctors import CBVCorrector

# Plotly for interactive periodogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Optional KvW implementation (PyAstronomy ≥0.20)
try:
    from PyAstronomy.pyasl import kweeVanWoerden
    HAS_KVW = True
except ImportError:
    HAS_KVW = False

# -----------------------------------------------------------------------------
# Helper routines
# -----------------------------------------------------------------------------

def sigma_clip(x: np.ndarray, sig: float = 4.0) -> np.ndarray:
    """
    Return boolean mask of points within *sig*×MAD from the median.
    
    Parameters:
    -----------
    x : np.ndarray
        Input data array
    sig : float
        Sigma clipping threshold (default: 4.0)
        
    Returns:
    --------
    np.ndarray
        Boolean mask of points within threshold
    """
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return np.abs(x - med) < sig * mad


def mask_outbursts(flux: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Mask outbursts in CV light curves using sigma clipping.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux measurements
    sigma : float
        Sigma clipping threshold (default: 4.0)
        
    Returns:
    --------
    np.ndarray
        Boolean mask of quiescent data points
    """
    quiet = sigma_clip(flux, sigma)
    expanded = quiet.copy()
    for i in range(1, len(flux) - 1):
        if not quiet[i]:
            expanded[i - 1 : i + 2] = False
    return expanded


def compute_oc(tmid: np.ndarray, t0: float, period: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute O-C residuals from eclipse mid-times.
    
    Parameters:
    -----------
    tmid : np.ndarray
        Eclipse mid-times (BJD_TDB)
    t0 : float
        Reference epoch (BJD_TDB)
    period : float
        Orbital period (days)
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Epochs and O-C residuals (days)
    """
    epochs = np.round((tmid - t0) / period).astype(int)
    calc = t0 + epochs * period
    return epochs, tmid - calc


# -----------------------------------------------------------------------------
# Template construction & timing measurement
# -----------------------------------------------------------------------------

def build_template(time: np.ndarray, flux: np.ndarray, period: float, nbins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Build eclipse template from folded light curve.
    
    Parameters:
    -----------
    time : np.ndarray
        Time array (BJD_TDB)
    flux : np.ndarray
        Flux array
    period : float
        Orbital period (days)
    nbins : int
        Number of phase bins (default: 200)
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Phase grid and normalized template flux
    """
    print(f"  • Folding light curve with period {period:.6f} days...")
    phase = ((time - time[0]) / period) % 1.0
    order = np.argsort(phase)
    binedges = np.linspace(0, 1, nbins + 1)
    digit = np.digitize(phase[order], binedges) - 1
    phase_bin = 0.5 * (binedges[:-1] + binedges[1:])
    print(f"  • Computing median in {nbins} phase bins...")
    template = np.array([np.nanmedian(flux[order][digit == i]) for i in range(nbins)])
    print(f"  • Normalizing template...")
    return phase_bin, template / np.nanmedian(template)


def measure_midpoints(time: np.ndarray, flux: np.ndarray, period: float, t0: float, 
                     pgrid: np.ndarray, tflux: np.ndarray, width: float = 0.2, 
                     output_dir: str = ".", save_examples: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure eclipse mid-times using template cross-correlation.
    
    Parameters:
    -----------
    time : np.ndarray
        Time array (BJD_TDB)
    flux : np.ndarray
        Flux array
    period : float
        Orbital period (days)
    t0 : float
        Reference epoch (BJD_TDB)
    pgrid : np.ndarray
        Phase grid for template
    tflux : np.ndarray
        Template flux
    width : float
        Eclipse window width in phase (default: 0.2)
    output_dir : str
        Output directory for example plots
    save_examples : bool
        Whether to save example eclipse plots
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Eclipse mid-times and uncertainties (BJD_TDB)
    """
    flux = flux / np.nanmedian(flux)
    mids, errs = [], []
    epochs = np.arange(np.floor((time.min() - t0) / period) - 1, np.ceil((time.max() - t0) / period) + 1).astype(int)
    print(f"  • Processing {len(epochs)} potential eclipse epochs...")
    
    valid_eclipses = 0
    example_count = 0
    max_examples = 5  # Save 5 random eclipses as examples
    
    # Create a list of valid eclipse indices for random selection
    valid_eclipse_indices = []
    
    for i, e in enumerate(epochs):
        if i % 10 == 0:
            print(f"    Processing epoch {e} ({i+1}/{len(epochs)})...")
        
        centre = t0 + e * period
        mask = np.abs(time - centre) < width * period / 2
        if mask.sum() < 10:
            continue
        
        valid_eclipses += 1
        valid_eclipse_indices.append((i, e, centre))
        
        seg_t = time[mask]
        seg_f = flux[mask]
        seg_phase = (seg_t - centre) / period + 0.5
        shifts = np.linspace(-0.02, 0.02, 201)
        chi2 = []
        for s in shifts:
            model = np.interp((seg_phase + s) % 1, pgrid, tflux, period=1)
            chi2.append(np.nansum((seg_f - model) ** 2))
        chi2 = np.array(chi2)
        best = shifts[np.argmin(chi2)]
        mids.append(centre + best * period)
        
        # σ ≈ sqrt(2/|d²χ²/dδ²|) scaled to phase; parabolic estimate
        try:
            a = np.polyfit(shifts[np.argmin(chi2) - 2 : np.argmin(chi2) + 3], chi2[np.argmin(chi2) - 2 : np.argmin(chi2) + 3], 2)[0]
            if a > 0:
                errs.append(np.sqrt(1 / a) * period)
            else:
                errs.append(0.0001 * period)
        except Exception:
            errs.append(0.0001 * period)
    
    # Randomly select examples from valid eclipses
    if save_examples and len(valid_eclipse_indices) > 0:
        random.shuffle(valid_eclipse_indices)
        selected_examples = valid_eclipse_indices[:min(max_examples, len(valid_eclipse_indices))]
        
        for example_count, (i, e, centre) in enumerate(selected_examples):
            # Create 1-day window around eclipse
            day_window = 1.0  # 1 day
            time_mask = np.abs(time - centre) < day_window / 2
            
            if np.sum(time_mask) > 50:  # Ensure we have enough data points
                window_time = time[time_mask]
                window_flux = flux[time_mask]
                
                # Create plot using matplotlib
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot the light curve
                ax.plot(window_time, window_flux, '.', markersize=2, alpha=0.7, color='blue', label='Data')
                
                # Mark the eclipse center
                ax.axvline(centre, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Eclipse center (Epoch {e})')
                
                # Mark the period boundaries around eclipse
                for offset in [-1, 0, 1]:
                    eclipse_time = centre + offset * period
                    if window_time.min() <= eclipse_time <= window_time.max():
                        ax.axvline(eclipse_time, color='orange', linestyle=':', alpha=0.6, linewidth=1)
                
                ax.set_xlabel('Time (BJD_TDB)')
                ax.set_ylabel('Normalized Flux')
                ax.set_title(f'Eclipse Example {example_count+1} - Epoch {e}\n1-day window around eclipse (P = {period:.6f} d)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add text box with eclipse info
                textstr = f'Eclipse depth: {(1 - np.min(window_flux[np.abs(window_time - centre) < 0.1 * period]))*100:.1f}%\nPeriod: {period*24:.2f} hours'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                example_file = os.path.join(output_dir, f"eclipse_example_{example_count+1}_epoch_{e}_1day.png")
                plt.savefig(example_file, dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"  • Successfully measured {valid_eclipses} eclipses out of {len(epochs)} candidates")
    if save_examples:
        print(f"  • Saved {min(max_examples, len(valid_eclipse_indices))} random eclipse example plots (1-day windows)")
    return np.array(mids), np.array(errs)


# -----------------------------------------------------------------------------
# Interactive periodogram functions
# -----------------------------------------------------------------------------

def create_interactive_periodogram(time_q: np.ndarray, flux_q: np.ndarray, freq: np.ndarray, 
                                 power: np.ndarray, method: str = "BLS") -> tuple[float, float]:
    """
    Create an interactive periodogram using Plotly that allows clicking to select period.
    
    Parameters:
    -----------
    time_q : np.ndarray
        Time array for quiescent data
    flux_q : np.ndarray
        Flux array for quiescent data
    freq : np.ndarray
        Frequency array from periodogram
    power : np.ndarray
        Power array from periodogram
    method : str
        Method used ("BLS" or "Lomb-Scargle")
    
    Returns:
    --------
    tuple[float, float]
        Selected period and corresponding T0 epoch
    """
    
    print("\n" + "=" * 50)
    print("INTERACTIVE PERIODOGRAM - Click to Select Period")
    print("=" * 50)
    
    # Convert frequency to period
    periods = 1.0 / freq
    
    # Create subplot with periodogram and folded light curve
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Periodogram - Click to Select Period', 'Folded Light Curve Preview'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1
    )
    
    # Add the periodogram trace
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=power,
            mode='lines',
            name=f'{method} Periodogram',
            line=dict(color='blue', width=1),
            hovertemplate='Period: %{x:.6f} days<br>Power: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Find and mark the highest peak
    max_power_idx = np.argmax(power)
    best_period = periods[max_power_idx]
    best_power = power[max_power_idx]
    
    fig.add_trace(
        go.Scatter(
            x=[best_period],
            y=[best_power],
            mode='markers',
            name='Highest Peak',
            marker=dict(color='red', size=10, symbol='star'),
            hovertemplate='Best Period: %{x:.6f} days<br>Power: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add folded light curve for the best period initially
    phase_best = ((time_q - time_q[0]) / best_period) % 1.0
    flux_norm = flux_q / np.nanmedian(flux_q)
    
    # Sort by phase for better plotting
    sort_idx = np.argsort(phase_best)
    phase_sorted = phase_best[sort_idx]
    flux_sorted = flux_norm[sort_idx]
    
    fig.add_trace(
        go.Scatter(
            x=phase_sorted,
            y=flux_sorted,
            mode='markers',
            name=f'Folded LC (P={best_period:.6f}d)',
            marker=dict(color='darkgreen', size=2, opacity=0.6),
            hovertemplate='Phase: %{x:.3f}<br>Flux: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add phase-binned version
    n_bins = 50
    bin_edges = np.linspace(0, 1, n_bins + 1)
    phase_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    flux_bins = []
    flux_errs = []
    for i in range(n_bins):
        mask = (phase_best >= bin_edges[i]) & (phase_best < bin_edges[i+1])
        if np.sum(mask) > 0:
            flux_bins.append(np.nanmedian(flux_norm[mask]))
            flux_errs.append(np.nanstd(flux_norm[mask]) / np.sqrt(np.sum(mask)))
        else:
            flux_bins.append(np.nan)
            flux_errs.append(np.nan)
    
    flux_bins = np.array(flux_bins)
    flux_errs = np.array(flux_errs)
    
    fig.add_trace(
        go.Scatter(
            x=phase_bins,
            y=flux_bins,
            error_y=dict(array=flux_errs, visible=True),
            mode='markers+lines',
            name='Binned LC',
            marker=dict(color='red', size=4),
            line=dict(color='red', width=2),
            hovertemplate='Phase: %{x:.3f}<br>Flux: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'Interactive {method} Periodogram - Click to Select Period',
        width=1000,
        height=800,
        hovermode='closest',
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Period (days)", range=[0.05, 0.3], row=1, col=1)
    fig.update_yaxes(title_text="Power", row=1, col=1)
    fig.update_xaxes(title_text="Phase", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Normalized Flux", row=2, col=1)
    
    # Add instructions
    fig.add_annotation(
        text="📍 Click on the periodogram to select your desired period",
        xref="paper", yref="paper",
        x=0.5, y=0.95,
        showarrow=False,
        font=dict(size=14, color="darkblue"),
        bgcolor="lightblue",
        bordercolor="blue",
        borderwidth=1
    )
    
    # Save the interactive plot
    html_file = "interactive_periodogram.html"
    fig.write_html(html_file)
    print(f"✓ Interactive periodogram saved to: {html_file}")
    print(f"✓ Opening in browser...")
    
    # Display the plot
    fig.show()
    
    print("\nINSTRUCTIONS:")
    print("1. The periodogram should open in your web browser")
    print("2. Click on the period you want to select")
    print("3. Note the period value from the hover tooltip")
    print("4. Return here and enter that period value")
    print("5. Or press Enter to use the highest peak automatically")
    
    # Get user input
    while True:
        try:
            user_input = input(f"\nEnter selected period (or press Enter for {best_period:.6f}): ").strip()
            
            if user_input == "":
                selected_period = best_period
                print(f"✓ Using highest peak: {selected_period:.6f} days")
                break
            else:
                selected_period = float(user_input)
                if 0.05 <= selected_period <= 0.3:
                    print(f"✓ Selected period: {selected_period:.6f} days")
                    break
                else:
                    print("⚠️  Period must be between 0.05 and 0.3 days. Please try again.")
        except ValueError:
            print("⚠️  Please enter a valid number or press Enter for default.")
    
    # If user selected a different period, show the folded light curve for that period
    if selected_period != best_period:
        print(f"\nGenerating folded light curve for selected period {selected_period:.6f} days...")
        
        # Create new folded light curve plot
        fig_folded = go.Figure()
        
        phase_selected = ((time_q - time_q[0]) / selected_period) % 1.0
        sort_idx = np.argsort(phase_selected)
        phase_sorted = phase_selected[sort_idx]
        flux_sorted = flux_norm[sort_idx]
        
        fig_folded.add_trace(
            go.Scatter(
                x=phase_sorted,
                y=flux_sorted,
                mode='markers',
                name=f'Folded LC (P={selected_period:.6f}d)',
                marker=dict(color='darkgreen', size=2, opacity=0.6),
                hovertemplate='Phase: %{x:.3f}<br>Flux: %{y:.4f}<extra></extra>'
            )
        )
        
        # Add phase-binned version for selected period
        flux_bins_sel = []
        flux_errs_sel = []
        for i in range(n_bins):
            mask = (phase_selected >= bin_edges[i]) & (phase_selected < bin_edges[i+1])
            if np.sum(mask) > 0:
                flux_bins_sel.append(np.nanmedian(flux_norm[mask]))
                flux_errs_sel.append(np.nanstd(flux_norm[mask]) / np.sqrt(np.sum(mask)))
            else:
                flux_bins_sel.append(np.nan)
                flux_errs_sel.append(np.nan)
        
        flux_bins_sel = np.array(flux_bins_sel)
        flux_errs_sel = np.array(flux_errs_sel)
        
        fig_folded.add_trace(
            go.Scatter(
                x=phase_bins,
                y=flux_bins_sel,
                error_y=dict(array=flux_errs_sel, visible=True),
                mode='markers+lines',
                name='Binned LC',
                marker=dict(color='red', size=4),
                line=dict(color='red', width=2),
                hovertemplate='Phase: %{x:.3f}<br>Flux: %{y:.4f}<extra></extra>'
            )
        )
        
        fig_folded.update_layout(
            title=f'Folded Light Curve for Selected Period: {selected_period:.6f} days',
            xaxis_title='Phase',
            yaxis_title='Normalized Flux',
            width=800,
            height=500,
            template='plotly_white'
        )
        
        fig_folded.update_xaxes(range=[0, 1])
        
        # Save and show the folded light curve
        html_file_folded = "selected_period_folded.html"
        fig_folded.write_html(html_file_folded)
        print(f"✓ Folded light curve saved to: {html_file_folded}")
        fig_folded.show()
    
    # Calculate T0 for the selected period
    selected_t0 = calculate_t0(time_q, flux_q, selected_period, method)
    
    return selected_period, selected_t0


def calculate_t0(time_q: np.ndarray, flux_q: np.ndarray, selected_period: float, method: str) -> float:
    """
    Calculate T0 epoch for the selected period.
    
    Parameters:
    -----------
    time_q : np.ndarray
        Time array for quiescent data
    flux_q : np.ndarray
        Flux array for quiescent data
    selected_period : float
        Selected orbital period (days)
    method : str
        Method used ("BLS" or "Lomb-Scargle")
        
    Returns:
    --------
    float
        T0 epoch (BJD_TDB)
    """
    print(f"\nCalculating T0 for selected period {selected_period:.6f} days...")
    
    # Manual phase folding and binning
    phase = ((time_q - time_q[0]) / selected_period) % 1.0
    
    # Create phase bins
    n_bins = 200
    bin_edges = np.linspace(0, 1, n_bins + 1)
    phase_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Bin the flux data
    flux_bins = []
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            flux_bins.append(np.nanmedian(flux_q[mask]))
        else:
            flux_bins.append(np.nan)
    
    flux_bins = np.array(flux_bins)
    
    if method == "BLS":
        # For BLS, find minimum (eclipse center)
        print("  • Estimating T0 from folded light curve...")
        primary_idx = np.nanargmin(flux_bins)
        primary_phase = phase_bins[primary_idx]
        
        # Set T0 based on primary eclipse phase
        selected_t0 = time_q[0] + primary_phase * selected_period
        
        print(f"  • Primary eclipse found at phase {primary_phase:.3f}")
        print(f"  • T0 = {selected_t0:.6f} BJD_TDB")
        
    else:  # Lomb-Scargle
        # Find local minima
        neg_flux = -flux_bins
        peaks, _ = find_peaks(neg_flux, height=np.percentile(neg_flux, 75))
        
        if len(peaks) >= 2:
            # Find the two deepest minima
            peak_depths = neg_flux[peaks]
            deepest_indices = peaks[np.argsort(peak_depths)[-2:]]
            
            # Check if they are roughly separated by 0.5 phase (for eclipsing binary)
            phases_deep = phase_bins[deepest_indices]
            phase_diff = abs(phases_deep[0] - phases_deep[1])
            if phase_diff > 0.5:
                phase_diff = 1.0 - phase_diff
            
            if 0.4 <= phase_diff <= 0.6:  # Roughly opposite phases
                # Choose the deeper one as primary
                primary_idx = deepest_indices[np.argmin(flux_bins[deepest_indices])]
                print(f"  • Two eclipses detected, selecting deeper one as primary")
            else:
                # Just use the deepest minimum
                primary_idx = peaks[np.argmax(peak_depths)]
                print(f"  • Single eclipse system detected")
        else:
            # Use the single deepest minimum
            primary_idx = np.nanargmin(flux_bins)
            print(f"  • Using deepest minimum as primary eclipse")
        
        primary_phase = phase_bins[primary_idx]
        
        # Ensure primary_phase is between 0 and 1
        primary_phase = primary_phase % 1.0
        
        # Set T0 based on primary eclipse phase
        selected_t0 = time_q[0] + primary_phase * selected_period
        
        # If primary eclipse is near phase 0.5, shift to center at phase 0
        if 0.45 <= primary_phase <= 0.55:
            selected_t0 -= 0.5 * selected_period
            primary_phase = (primary_phase - 0.5) % 1.0
            print(f"  • Primary eclipse at phase ≈0.5, shifting epoch by -0.5P")
        
        print(f"  • Primary eclipse found at phase {primary_phase:.3f}")
        print(f"  • T0 = {selected_t0:.6f} BJD_TDB")
    
    return selected_t0


# -----------------------------------------------------------------------------
# Data loading and processing functions
# -----------------------------------------------------------------------------

def load_tess_data(tic_id: int, sector: int = None, download_number: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Load TESS data for a given TIC ID.
    
    Parameters:
    -----------
    tic_id : int
        TIC identifier
    sector : int, optional
        Specific sector to download (None for all)
    download_number : int
        Maximum number of sectors to download
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Time and flux arrays
    """
    print(f"Using TIC {tic_id}, sector {sector if sector else 'All available'}")
    print("Searching for TESS data...")
    
    # Use modern search_lightcurve with author="SPOC"
    sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector)
    if len(sr) == 0:
        print("❌ ERROR: No SPOC light curves found!")
        raise SystemExit(f"No SPOC light curves found for TIC {tic_id}, sector {sector}")
    
    print(f"✓ Found {len(sr)} light curve(s)")
    print("Downloading light curve data...")
    
    lcc = sr.download_all()
    if download_number is not None:
        print(f"✓ Downloading first {download_number} sectors")
        lc = lcc[:download_number].stitch()
    else:
        lc = lcc.stitch()

    print(f"✓ Downloaded successfully")
    print(f"  Data points: {len(lc)}")
    print(f"  Time range: {lc.time.value.min():.2f} - {lc.time.value.max():.2f} BJD_TDB")
    print(f"  Duration: {(lc.time.value.max() - lc.time.value.min()):.2f} days")
    
    # Ensure PDCSAP flux (already detrended)
    time = lc.time.value  # BJD_TDB
    flux = lc.flux.value
    print(f"✓ Using PDCSAP flux column")
    
    return time, flux


def load_file_data(filepath: str, time_column: str = "time", flux_column: str = "flux") -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV or FITS file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    time_column : str
        Name of time column
    flux_column : str
        Name of flux column
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Time and flux arrays
    """
    print(f"Loading file: {filepath}")
    
    path = Path(filepath)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        print("✓ CSV file loaded")
    else:
        df = lk.read(path).to_pandas()
        print("✓ FITS file loaded and converted")
        
    time = df[time_column].values
    flux = df[flux_column].values
    print(f"✓ Extracted columns: {time_column}, {flux_column}")
    print(f"  Data points: {len(time)}")
    print(f"  Time range: {time.min():.2f} - {time.max():.2f}")
    print(f"  Duration: {(time.max() - time.min()):.2f} days")
    
    return time, flux


# -----------------------------------------------------------------------------
# Analysis and plotting functions
# -----------------------------------------------------------------------------

def create_diagnostic_plots(time: np.ndarray, flux: np.ndarray, time_q: np.ndarray, 
                           flux_q: np.ndarray, period: float, t0: float, 
                           output_file: str) -> dict:
    """
    Create diagnostic plots for the analysis.
    
    Parameters:
    -----------
    time : np.ndarray
        Full time array
    flux : np.ndarray
        Full flux array
    time_q : np.ndarray
        Quiescent time array
    flux_q : np.ndarray
        Quiescent flux array
    period : float
        Orbital period
    t0 : float
        Reference epoch
    output_file : str
        Output filename
        
    Returns:
    --------
    dict
        Dictionary with diagnostic statistics
    """
    print("Creating diagnostic plots...")
    
    # Plot 1: Raw lightcurve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(time, flux, 'k.', markersize=1, alpha=0.5)
    plt.plot(time_q, flux_q, 'b.', markersize=1, alpha=0.7)
    plt.xlabel('Time (BJD_TDB)')
    plt.ylabel('Flux')
    plt.title('Raw Light Curve')
    plt.legend(['All data', 'Quiescent data'])
    
    # Plot 2: Folded lightcurve
    plt.subplot(2, 2, 2)
    phase = ((time_q - t0) / period) % 1.0
    flux_norm = flux_q / np.nanmedian(flux_q)
    plt.plot(phase, flux_norm, 'b.', markersize=2, alpha=0.6)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.title(f'Folded Light Curve (P={period:.6f}d)')
    plt.xlim(0, 1)
    
    # Plot 3: Phase-binned lightcurve
    plt.subplot(2, 2, 3)
    bins = np.linspace(0, 1, 50)
    digitized = np.digitize(phase, bins) - 1
    phase_bin = 0.5 * (bins[:-1] + bins[1:])
    flux_bin = np.array([np.nanmedian(flux_norm[digitized == i]) for i in range(len(bins)-1)])
    flux_err = np.array([np.nanstd(flux_norm[digitized == i])/np.sqrt(np.sum(digitized == i)) 
                        for i in range(len(bins)-1)])
    
    plt.errorbar(phase_bin, flux_bin, yerr=flux_err, fmt='ro-', markersize=4, capsize=2)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.title('Phase-Binned Light Curve')
    plt.xlim(0, 1)
    
    # Plot 4: Eclipse depth analysis
    plt.subplot(2, 2, 4)
    eclipse_depth = 1 - np.min(flux_bin)
    plt.bar(['Eclipse Depth'], [eclipse_depth * 100])
    plt.ylabel('Eclipse Depth (%)')
    plt.title(f'Eclipse Analysis\nDepth: {eclipse_depth*100:.2f}%')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Diagnostic plots saved to: {output_file}")
    plt.close()
    
    # Calculate and return statistics
    stats = {
        'flux_range': (np.min(flux_norm), np.max(flux_norm)),
        'flux_std': np.std(flux_norm),
        'min_flux_folded': np.min(flux_bin),
        'eclipse_depth': eclipse_depth,
        'phase_of_minimum': phase_bin[np.argmin(flux_bin)]
    }
    
    print(f"\nDiagnostic Statistics:")
    print(f"  Flux range: {stats['flux_range'][0]:.4f} - {stats['flux_range'][1]:.4f}")
    print(f"  Flux std: {stats['flux_std']:.4f}")
    print(f"  Minimum flux in folded curve: {stats['min_flux_folded']:.4f}")
    print(f"  Estimated eclipse depth: {stats['eclipse_depth']*100:.2f}%")
    print(f"  Phase of minimum: {stats['phase_of_minimum']:.3f}")

    if eclipse_depth < 0.05:  # Less than 5% depth
        print("⚠️  WARNING: Eclipse depth is very shallow (<5%)")
        print("   This might explain why no eclipses were detected")
        print("   Consider adjusting detection thresholds or using a different target")
    
    return stats


def create_oc_plot(epochs: np.ndarray, oc: np.ndarray, terr: np.ndarray, 
                   coeff: np.ndarray, output_file: str) -> None:
    """
    Create O-C diagram plot.
    
    Parameters:
    -----------
    epochs : np.ndarray
        Epoch numbers
    oc : np.ndarray
        O-C residuals (days)
    terr : np.ndarray
        Timing uncertainties (days)
    coeff : np.ndarray
        Linear fit coefficients
    output_file : str
        Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(epochs, oc * 86400, yerr=terr * 86400, fmt="ko", ms=4, capsize=3, alpha=0.7)
    plt.plot(epochs, (coeff[0] * epochs + coeff[1]) * 86400, "r--", lw=2, label="Linear fit")
    plt.xlabel("Cycle number", fontsize=12)
    plt.ylabel("O – C (seconds)", fontsize=12)
    plt.title("O-C Diagram", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180, bbox_inches='tight')
    print(f"✓ O-C plot saved to: {output_file}")
    plt.close()


def save_analysis_report(period: float, period_new: float, t0: float, t0_new: float, 
                        tmid: np.ndarray, coeff: np.ndarray, quad_coeff: np.ndarray, 
                        output_file: str) -> None:
    """
    Save comprehensive analysis report.
    
    Parameters:
    -----------
    period : float
        Initial period
    period_new : float
        Refined period
    t0 : float
        Initial T0
    t0_new : float
        Refined T0
    tmid : np.ndarray
        Eclipse mid-times
    coeff : np.ndarray
        Linear fit coefficients
    quad_coeff : np.ndarray
        Quadratic fit coefficients
    output_file : str
        Output filename
    """
    # Calculate period change rates
    dPdt_sec_per_cycle = 2 * quad_coeff[0]  # dP/dt in seconds per cycle
    dPdt_sec_per_year = dPdt_sec_per_cycle / period * 365.25  # Convert to seconds per year
    
    with open(output_file, "w") as fh:
        fh.write("CV Eclipse Timing & O-C Analysis Report\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        fh.write("EPHEMERIS RESULTS:\n")
        fh.write("-" * 20 + "\n")
        fh.write(f"Initial period:  {period:.10f} d\n")
        fh.write(f"Refined period:  {period_new:.10f} d\n")
        fh.write(f"Initial T0:      {t0:.10f} BJD_TDB\n")
        fh.write(f"Refined T0:      {t0_new:.10f} BJD_TDB\n")
        fh.write(f"N mid-times:     {len(tmid)}\n\n")
        
        fh.write("LINEAR O-C FIT:\n")
        fh.write("-" * 15 + "\n")
        fh.write(f"Linear slope:    {coeff[0] * 86400:.3f} s/cycle\n")
        fh.write(f"Linear offset:   {coeff[1] * 86400:.1f} s\n\n")
        
        fh.write("QUADRATIC O-C FIT:\n")
        fh.write("-" * 18 + "\n")
        fh.write(f"Quadratic coeff (c): {quad_coeff[0]:.2e} s/cycle²\n")
        fh.write(f"Linear coeff (b):    {quad_coeff[1]:.3f} s/cycle\n")
        fh.write(f"Constant term (a):   {quad_coeff[2]:.1f} s\n")
        fh.write(f"Period change rate:  {dPdt_sec_per_cycle:.2e} s/cycle\n")
        fh.write(f"Period change rate:  {dPdt_sec_per_year:.2e} s/year\n\n")
        
        fh.write("TIMING STATISTICS:\n")
        fh.write("-" * 18 + "\n")
        if len(tmid) > 0:
            fh.write(f"Time span:       {(tmid.max() - tmid.min()):.2f} days\n")
            fh.write(f"Number of cycles: {len(tmid)}\n")
        fh.write(f"Period (hours):  {period_new * 24:.4f} h\n")
        fh.write(f"Period (minutes): {period_new * 24 * 60:.2f} min\n")
    
    print(f"✓ Analysis report saved to: {output_file}")

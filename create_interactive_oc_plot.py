#!/usr/bin/env python3.11
"""
Interactive O-C Plot Generator for CV Eclipse Timing Analysis

This script creates interactive HTML plots from the output of cv_eclipse_oc_pipeline.py
using Plotly. It reads the analysis results and generates interactive O-C diagrams.

O-C (Observed minus Calculated) diagrams are fundamental tools in eclipse timing
analysis of binary stars and cataclysmic variables. They show deviations from
a constant orbital period and can reveal:
- Period changes due to mass transfer
- Gravitational wave radiation
- Apsidal motion
- Third body effects

Requirements:
    pip install plotly>=5 pandas numpy

Usage:
    python create_interactive_oc_plot.py <output_prefix>

Example:
    python create_interactive_oc_plot.py ran_at_20250711_233623/final_test

This will generate:
    <output_prefix>_OC_interactive.html
"""

# Standard library imports for system operations and argument parsing
import argparse
import sys
import os

# Scientific computing libraries
import numpy as np          # Numerical operations and arrays
import pandas as pd         # Data manipulation and CSV reading
import plotly.graph_objects as go  # Interactive plotting library

# Path handling utility
from pathlib import Path

def load_results(output_prefix):
    """
    Load the analysis results from the CV pipeline output files.
    
    This function reads two critical files:
    1. _midtimes.csv: Contains eclipse mid-times and their uncertainties
    2. _report.txt: Contains fitted parameters (period, T0, slopes, etc.)
    
    Args:
        output_prefix (str): Base filename prefix (without extension)
                           e.g., "ran_at_20250711_233623/final_test"
    
    Returns:
        dict: Dictionary containing all loaded parameters and data arrays
    
    Raises:
        FileNotFoundError: If required input files don't exist
        ValueError: If essential parameters can't be parsed from report
    """
    
    # Construct expected filenames based on the prefix
    midtimes_file = output_prefix + "_midtimes.csv"
    report_file = output_prefix + "_report.txt"
    
    # Verify that both required files exist before proceeding
    if not os.path.exists(midtimes_file):
        raise FileNotFoundError(f"Could not find {midtimes_file}")
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"Could not find {report_file}")
    
    # Load eclipse mid-times data from CSV file
    # Expected columns: BJD_TDB (Barycentric Julian Date), err_d (error in days)
    print(f"Loading eclipse mid-times from {midtimes_file}...")
    df = pd.read_csv(midtimes_file)
    tmid = df['BJD_TDB'].values  # Eclipse mid-times in BJD_TDB
    terr = df['err_d'].values    # Timing uncertainties in days
    
    # Parse the analysis report file to extract fitted parameters
    print(f"Loading analysis parameters from {report_file}...")
    with open(report_file, 'r') as f:
        lines = f.readlines()
    
    # Initialize parameter variables to None
    # These will be filled by parsing the report file
    period = None        # Orbital period in days
    t0 = None           # Reference epoch (T0) in BJD_TDB
    linear_slope = None  # Linear trend in O-C (seconds per cycle)
    linear_offset = None # Linear offset (seconds)
    quad_a = None       # Quadratic fit: constant term
    quad_b = None       # Quadratic fit: linear coefficient
    quad_c = None       # Quadratic fit: quadratic coefficient
    
    # Parse each line of the report file to extract numerical parameters
    # The report file contains formatted output with parameter names and values
    for line in lines:
        # Extract initial orbital period (in days)
        if line.startswith("Initial period:"):
            period = float(line.split()[-2])  # Second-to-last word is the value
        # Extract reference epoch T0 (in BJD_TDB)
        elif line.startswith("Initial T0:"):
            t0 = float(line.split()[-2])
        # Extract linear trend slope (seconds per cycle)
        elif line.startswith("Linear slope:"):
            linear_slope = float(line.split()[-2])
        # Extract linear trend offset (seconds)
        elif line.startswith("Linear offset:"):
            linear_offset = float(line.split()[-2])
        # Extract quadratic fit coefficients
        # Note: These follow polynomial convention: f(x) = ax² + bx + c
        elif line.startswith("Quadratic coeff (c):"):
            quad_c = float(line.split()[-2])  # Quadratic coefficient (x²)
        elif line.startswith("Linear coeff (b):"):
            quad_b = float(line.split()[-2])  # Linear coefficient (x)
        elif line.startswith("Constant term (a):"):
            quad_a = float(line.split()[-2])  # Constant term
    
    # Validate that essential parameters were successfully parsed
    if period is None or t0 is None:
        raise ValueError("Could not parse period and T0 from report file")
    
    # Return all loaded parameters in a dictionary for easy access
    return {
        'tmid': tmid,              # Eclipse mid-times array
        'terr': terr,              # Timing uncertainty array
        'period': period,          # Orbital period (days)
        't0': t0,                  # Reference epoch (BJD_TDB)
        'linear_slope': linear_slope,     # Linear trend slope (s/cycle)
        'linear_offset': linear_offset,   # Linear trend offset (s)
        'quad_a': quad_a,          # Quadratic fit constant term
        'quad_b': quad_b,          # Quadratic fit linear coefficient
        'quad_c': quad_c           # Quadratic fit quadratic coefficient
    }

def compute_oc(tmid, t0, period):
    """
    Compute O-C (Observed minus Calculated) residuals for eclipse timings.
    
    O-C analysis is the foundation of eclipse timing studies. For each observed
    eclipse time, we calculate what the time "should" have been based on a
    constant period, then find the difference.
    
    The calculation process:
    1. Determine cycle number (epoch) for each eclipse
    2. Calculate expected time using: T_calc = T0 + E × P
    3. Compute residual: O-C = T_observed - T_calculated
    
    Args:
        tmid (array): Observed eclipse mid-times in BJD_TDB
        t0 (float): Reference epoch (time of cycle 0) in BJD_TDB
        period (float): Orbital period in days
    
    Returns:
        tuple: (epochs, oc_residuals)
            - epochs: Integer cycle numbers for each eclipse
            - oc_residuals: O-C values in days
    """
    # Calculate cycle numbers (epochs) for each observed eclipse
    # Round to nearest integer since we expect integer cycle numbers
    epochs = np.round((tmid - t0) / period).astype(int)
    
    # Calculate what the eclipse times should be for a constant period
    calc = t0 + epochs * period
    
    # Compute O-C residuals: Observed - Calculated
    # Positive values mean eclipses occur later than expected
    # Negative values mean eclipses occur earlier than expected
    oc = tmid - calc
    
    return epochs, oc

def create_interactive_plot(results, output_file):
    """
    Create an interactive Plotly O-C diagram with fitted curves.
    
    This function generates a publication-quality interactive plot showing:
    - Observed O-C data points with error bars
    - Linear trend fit (if period is changing linearly)
    - Quadratic fit (if period changes are accelerating/decelerating)
    - Hover information and zoom capabilities
    - Parameter annotations
    
    Args:
        results (dict): Dictionary containing all analysis results from load_results()
        output_file (str): Path where HTML file will be saved
    
    Returns:
        plotly.graph_objects.Figure: The created Plotly figure object
    """
    
    # Compute O-C residuals from the observed eclipse times
    epochs, oc = compute_oc(results['tmid'], results['t0'], results['period'])
    
    # Convert time units from days to seconds for better readability
    # (O-C values are typically small fractions of a day)
    oc_sec = oc * 86400  # Convert days to seconds (86400 sec/day)
    terr_sec = results['terr'] * 86400  # Convert error bars to seconds
    
    # Create a smooth grid of epoch values for plotting fitted curves
    # Use more points than data for smooth curve appearance
    e_grid = np.linspace(epochs.min(), epochs.max(), 500)
    
    # Calculate linear trend curve if parameters are available
    # Convert slope from s/cycle to days/cycle for calculation, then back to seconds
    if results['linear_slope'] is not None and results['linear_offset'] is not None:
        linear_slope_days = results['linear_slope'] / 86400  # Convert to days/cycle
        linear_offset_days = results['linear_offset'] / 86400  # Convert to days
        # Calculate linear trend: O-C = slope × epoch + offset
        linear_curve = (linear_slope_days * e_grid + linear_offset_days) * 86400  # Back to seconds
    else:
        linear_curve = None
    
    # Calculate quadratic fit curve if coefficients are available
    # Quadratic model: O-C = a + b×E + c×E²
    if (results['quad_a'] is not None and 
        results['quad_b'] is not None and 
        results['quad_c'] is not None):
        # Arrange coefficients for numpy.polyval: [highest_degree, ..., constant]
        quad_coeff = [results['quad_c'], results['quad_b'], results['quad_a']]  # [c, b, a]
        quad_curve = np.polyval(quad_coeff, e_grid)
    else:
        quad_curve = None
    
    # Create the main Plotly figure object
    fig = go.Figure()
    
    # Add observed O-C data points with error bars
    # These are the actual measurements that everything else is fitted to
    fig.add_trace(go.Scatter(
        x=epochs,  # X-axis: cycle numbers (integer values)
        y=oc_sec,  # Y-axis: O-C residuals in seconds
        mode='markers',  # Show only points, no connecting lines
        error_y=dict(
            type='data', 
            array=terr_sec, 
            visible=True,
            color='black',
            thickness=1
        ),
        name='O-C data',  # Legend label
        marker=dict(color='black', size=5, symbol='circle'),
        # Custom hover template for detailed information on mouse-over
        hovertemplate='<b>Eclipse Data</b><br>' +
                      'Epoch: %{x}<br>' +
                      'O-C: %{y:.2f} s<br>' +
                      'Error: %{error_y.array:.2f} s<extra></extra>'
    ))
    
    # Add linear trend fit curve if available
    if linear_curve is not None:
        fig.add_trace(go.Scatter(
            x=e_grid,
            y=linear_curve,
            mode='lines',  # Show as continuous line
            name='Linear fit',  # Legend label
            line=dict(dash='dash', color='red', width=2),  # Red dashed line
            hovertemplate='<b>Linear Trend</b><br>' +
                          'Epoch: %{x:.1f}<br>' +
                          'Linear fit: %{y:.2f} s<extra></extra>'
        ))
    
    # Add quadratic fit curve if available
    # Quadratic fits can reveal period acceleration/deceleration
    if quad_curve is not None:
        fig.add_trace(go.Scatter(
            x=e_grid,
            y=quad_curve,
            mode='lines',
            name='Quadratic fit',  # Legend label
            line=dict(color='blue', width=3),  # Solid blue line, thicker than linear
            hovertemplate='<b>Quadratic Fit</b><br>' +
                          'Epoch: %{x:.1f}<br>' +
                          'Quadratic fit: %{y:.2f} s<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='Cycle number',
        yaxis_title='O – C (s)',
        title='O-C diagram (interactive)',
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        width=900,
        height=600
    )
    
    # Add annotations with fit parameters
    annotation_text = f"Period: {results['period']:.8f} days<br>"
    annotation_text += f"T0: {results['t0']:.6f} BJD_TDB<br>"
    if results['linear_slope'] is not None:
        annotation_text += f"Linear slope: {results['linear_slope']:.3f} s/cycle<br>"
    if results['quad_c'] is not None:
        period_change_rate = 2 * results['quad_c'] / results['period'] * 365.25
        annotation_text += f"Period change: {period_change_rate:.2e} s/year"
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=annotation_text,
        showarrow=False,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=10)
    )
    
    # Save HTML file
    fig.write_html(output_file)
    print(f"✓ Interactive O-C diagram saved to: {output_file}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Create interactive O-C plot from CV pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "output_prefix",
        help="Output prefix from cv_eclipse_oc_pipeline.py (e.g., 'ran_at_20250711_233623/final_test')"
    )
    parser.add_argument(
        "--output",
        help="Output HTML filename (default: <output_prefix>_OC_interactive.html)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load analysis results
        print("=" * 60)
        print("Interactive O-C Plot Generator")
        print("=" * 60)
        
        results = load_results(args.output_prefix)
        
        print(f"✓ Loaded {len(results['tmid'])} eclipse timings")
        print(f"  Period: {results['period']:.8f} days")
        print(f"  T0: {results['t0']:.6f} BJD_TDB")
        
        # Determine output filename
        if args.output:
            output_file = args.output
        else:
            output_file = args.output_prefix + "_OC_interactive.html"
        
        # Create interactive plot
        print(f"\nCreating interactive O-C diagram...")
        fig = create_interactive_plot(results, output_file)
        
        print(f"\n✓ Interactive plot creation complete!")
        print(f"  Open {output_file} in your web browser to view the interactive plot")
        
        # Calculate some statistics
        epochs, oc = compute_oc(results['tmid'], results['t0'], results['period'])
        oc_sec = oc * 86400
        print(f"\nO-C Statistics:")
        print(f"  RMS scatter: {np.std(oc_sec):.1f} seconds")
        print(f"  Range: {oc_sec.min():.1f} to {oc_sec.max():.1f} seconds")
        print(f"  Number of eclipses: {len(epochs)}")
        print(f"  Epoch range: {epochs.min()} to {epochs.max()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

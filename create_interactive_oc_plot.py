#!/usr/bin/env python3.11
"""
Interactive O-C Plot Generator for CV Eclipse Timing Analysis

This script creates interactive HTML plots from the output of cv_eclipse_oc_pipeline.py
using Plotly. It reads the analysis results and generates interactive O-C diagrams.

Requirements:
    pip install plotly>=5 pandas numpy

Usage:
    python create_interactive_oc_plot.py <output_prefix>

Example:
    python create_interactive_oc_plot.py ran_at_20250711_233623/final_test

This will generate:
    <output_prefix>_OC_interactive.html
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def load_results(output_prefix):
    """Load the analysis results from the CV pipeline output files."""
    
    # Check if files exist
    midtimes_file = output_prefix + "_midtimes.csv"
    report_file = output_prefix + "_report.txt"
    
    if not os.path.exists(midtimes_file):
        raise FileNotFoundError(f"Could not find {midtimes_file}")
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"Could not find {report_file}")
    
    # Load mid-times data
    print(f"Loading eclipse mid-times from {midtimes_file}...")
    df = pd.read_csv(midtimes_file)
    tmid = df['BJD_TDB'].values
    terr = df['err_d'].values
    
    # Parse report file to get period and T0
    print(f"Loading analysis parameters from {report_file}...")
    with open(report_file, 'r') as f:
        lines = f.readlines()
    
    # Extract parameters
    period = None
    t0 = None
    linear_slope = None
    linear_offset = None
    quad_a = None
    quad_b = None
    quad_c = None
    
    for line in lines:
        if line.startswith("Initial period:"):
            period = float(line.split()[-2])
        elif line.startswith("Initial T0:"):
            t0 = float(line.split()[-2])
        elif line.startswith("Linear slope:"):
            linear_slope = float(line.split()[-2])
        elif line.startswith("Linear offset:"):
            linear_offset = float(line.split()[-2])
        elif line.startswith("Quadratic coeff (c):"):
            quad_c = float(line.split()[-2])
        elif line.startswith("Linear coeff (b):"):
            quad_b = float(line.split()[-2])
        elif line.startswith("Constant term (a):"):
            quad_a = float(line.split()[-2])
    
    if period is None or t0 is None:
        raise ValueError("Could not parse period and T0 from report file")
    
    return {
        'tmid': tmid,
        'terr': terr,
        'period': period,
        't0': t0,
        'linear_slope': linear_slope,
        'linear_offset': linear_offset,
        'quad_a': quad_a,
        'quad_b': quad_b,
        'quad_c': quad_c
    }

def compute_oc(tmid, t0, period):
    """Compute O-C residuals."""
    epochs = np.round((tmid - t0) / period).astype(int)
    calc = t0 + epochs * period
    oc = tmid - calc
    return epochs, oc

def create_interactive_plot(results, output_file):
    """Create interactive Plotly O-C diagram."""
    
    # Compute O-C residuals
    epochs, oc = compute_oc(results['tmid'], results['t0'], results['period'])
    oc_sec = oc * 86400  # Convert to seconds
    terr_sec = results['terr'] * 86400  # Convert errors to seconds
    
    # Create epoch grid for smooth curves
    e_grid = np.linspace(epochs.min(), epochs.max(), 500)
    
    # Linear fit curve (convert slope from s/cycle to days/cycle for calculation)
    linear_slope_days = results['linear_slope'] / 86400  # Convert back to days/cycle
    linear_offset_days = results['linear_offset'] / 86400  # Convert back to days
    linear_curve = (linear_slope_days * e_grid + linear_offset_days) * 86400  # Convert to seconds
    
    # Quadratic fit curve (coefficients are already in seconds)
    if results['quad_a'] is not None and results['quad_b'] is not None and results['quad_c'] is not None:
        quad_coeff = [results['quad_c'], results['quad_b'], results['quad_a']]  # [c, b, a] format
        quad_curve = np.polyval(quad_coeff, e_grid)
    else:
        quad_curve = None
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add data points with error bars
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=oc_sec,
        mode='markers',
        error_y=dict(type='data', array=terr_sec, visible=True),
        name='O-C data',
        marker=dict(color='black', size=5),
        hovertemplate='Epoch: %{x}<br>O-C: %{y:.2f} s<br>Error: %{error_y.array:.2f} s<extra></extra>'
    ))
    
    # Add linear fit
    fig.add_trace(go.Scatter(
        x=e_grid,
        y=linear_curve,
        mode='lines',
        name='Linear fit',
        line=dict(dash='dash', color='red', width=2),
        hovertemplate='Epoch: %{x}<br>Linear fit: %{y:.2f} s<extra></extra>'
    ))
    
    # Add quadratic fit if available
    if quad_curve is not None:
        fig.add_trace(go.Scatter(
            x=e_grid,
            y=quad_curve,
            mode='lines',
            name='Quadratic fit',
            line=dict(color='blue', width=3),
            hovertemplate='Epoch: %{x}<br>Quadratic fit: %{y:.2f} s<extra></extra>'
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

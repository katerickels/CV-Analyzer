# CV-Analyzer

A comprehensive toolkit for analyzing Cataclysmic Variable (CV) stars using astronomical data from various sources including TESS, Kepler, and AAVSO.

## Features

- **Light Curve Analysis**: Process and analyze light curves from TESS, Kepler, and other sources
- **Periodogram Analysis**: Gaussian fitting and period detection
- **O-C Diagram Generation**: Orbital period analysis and timing variations
- **Eclipse Analysis**: Automated eclipse detection and timing
- **Data Visualization**: Interactive plots using matplotlib, seaborn, and plotly
- **Outburst Detection**: Automated detection and removal of outburst events

## Project Structure

```
CV-Analyzer/
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore              # Git ignore patterns
├── venv/                   # Virtual environment (created after setup)
├── New Data/              # Current analysis notebooks and data
├── Refactored/            # Cleaned up analysis modules
├── Future Projects/       # Planned analysis projects
├── Old Data/             # Archive of previous analyses
└── Temporary Files/      # Working files and experiments
```

## Setup and Installation

### Prerequisites

- Python 3.11
- Git (for cloning the repository)

### Quick Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/katerickels/CV-Analyzer.git
   cd CV-Analyzer
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python3.11 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Verification

To verify your installation works correctly:

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Test import of key packages
python -c "import lightkurve, numpy, pandas, matplotlib; print('All packages imported successfully!')"
```

## Usage

### Working with Virtual Environment

**Always activate the virtual environment before working:**

```bash
# Navigate to project directory
cd /path/to/CV-Analyzer

# Activate virtual environment
source venv/bin/activate

# Your command prompt should now show (venv)
# Now you can run Python scripts or start Jupyter
```

**When you're done working:**

```bash
# Deactivate virtual environment
deactivate
```

### Running Jupyter Notebooks

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```
   
   Or for JupyterLab:
   ```bash
   jupyter lab
   ```

3. **Navigate to your analysis notebooks** in the `New Data/` or `Refactored/` directories.

### Key Analysis Notebooks

- **`New Data/Light Curve and Periodogram.ipynb`**: Main light curve analysis
- **`New Data/O-C maker v2.ipynb`**: Generate O-C diagrams
- **`Refactored/OC_Analyser_v5.ipynb`**: Comprehensive CV analysis pipeline
- **`New Data/Gaussian fit in Periodigram.ipynb`**: Period detection with Gaussian fitting

## Key Dependencies

- **lightkurve**: TESS/Kepler data access and analysis
- **astropy**: Astronomical calculations and data handling
- **numpy/pandas**: Data manipulation and analysis
- **matplotlib/seaborn/plotly**: Data visualization
- **scipy**: Scientific computing and fitting
- **astroquery**: Query astronomical databases

## Data Sources

This project works with data from:
- **TESS**: Transiting Exoplanet Survey Satellite
- **Kepler**: Kepler Space Telescope
- **AAVSO**: American Association of Variable Star Observers
- **ASAS-SN**: All-Sky Automated Survey for Supernovae

## Common Workflows

### 1. Analyzing a New CV
1. Download light curve data using lightkurve
2. Process and clean the data
3. Generate periodograms for period detection
4. Create phase-folded light curves
5. Identify and time eclipse events
6. Generate O-C diagrams for period analysis

### 2. Batch Processing
1. Use the modules in `Refactored/modules.py`
2. Run the comprehensive analysis pipeline
3. Generate diagnostic plots and reports

## Troubleshooting

### Virtual Environment Issues

**If you get "command not found" errors:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
```

**If packages seem missing:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

**If you get permission errors:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### Common Import Errors

**If lightkurve fails to import:**
```bash
# Try updating
pip install --upgrade lightkurve
```

**If matplotlib plots don't display:**
```bash
# For Jupyter notebooks
%matplotlib inline
```

## Contributing

1. Create a new branch for your analysis
2. Keep notebooks organized in appropriate directories
3. Update requirements.txt if you add new dependencies
4. Document your analysis methods clearly

## Data Management

- Large data files (> 100MB) should not be committed to git
- Use the `.gitignore` file to exclude temporary analysis outputs
- Store processed data in appropriate subdirectories
- Consider using `ran_at_*` directories for time-stamped analysis runs

## Support

For questions about:
- **Astronomical analysis**: Consult the lightkurve documentation
- **Data processing**: Check pandas and numpy documentation  
- **Plotting**: Refer to matplotlib/seaborn/plotly documentation
- **Project-specific issues**: Create an issue in the repository

## License

This project is for academic and research purposes. Please cite appropriate sources when using this code for publications.

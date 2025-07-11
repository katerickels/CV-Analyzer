import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def jd_to_tdb(jd_utc):
    """
    Convert Julian Date from Universal Time (UTC) to Barycentric Dynamical Time (TDB)
    
    This function applies corrections to convert from Earth-based time (UTC) to 
    time referenced to the solar system's center of mass (barycenter).
    
    Args:
        jd_utc (float or array): Julian Date in UTC
        
    Returns:
        float or array: Julian Date in TDB (Barycentric Dynamical Time)
    """
    # Convert to numpy array for vectorized operations
    jd_utc = np.asarray(jd_utc)
    
    # Time difference from J2000.0 in centuries
    T = (jd_utc - 2451545.0) / 36525.0
    
    # Conversion from UTC to TT (Terrestrial Time)
    # TAI-UTC leap seconds approximation (roughly 37 seconds as of 2020s)
    # TT = TAI + 32.184 seconds, so TT - UTC ≈ 69 seconds
    tt_utc_seconds = 69.184  # This should be updated based on current leap seconds
    jd_tt = jd_utc + tt_utc_seconds / 86400.0
    
    # Conversion from TT to TDB (Barycentric Dynamical Time)
    # This is a simplified formula for the periodic correction
    # Full calculation would require planetary ephemeris data
    
    # Mean longitude of the Sun (degrees)
    L = 280.4665 + 36000.7698 * T
    
    # Mean anomaly of the Sun (degrees)  
    g = 357.5291 + 35999.0503 * T
    
    # Convert to radians
    L_rad = np.radians(L % 360)
    g_rad = np.radians(g % 360)
    
    # Periodic correction in seconds (simplified formula)
    # This accounts for Earth's elliptical orbit and other effects
    correction_seconds = (0.001658 * np.sin(g_rad) + 
                         0.000014 * np.sin(2 * g_rad) -
                         0.000005 * np.cos(2 * L_rad))
    
    # Convert correction to days and add to TT
    jd_tdb = jd_tt + correction_seconds / 86400.0
    
    return jd_tdb

def remove_non_numeric_magnitude(df, mag_col='Magnitude'):
    """
    Remove rows where the magnitude column is not a float or integer.
    
    Args:
        df (pd.DataFrame): DataFrame containing a magnitude column
        mag_col (str): Name of the magnitude column (default 'Magnitude')
        
    Returns:
        pd.DataFrame: DataFrame with only numeric magnitude values
    """
    # Convert to numeric, setting errors to NaN, then drop NaN
    df[mag_col] = pd.to_numeric(df[mag_col], errors='coerce')
    cleaned_df = df.dropna(subset=[mag_col])
    return cleaned_df

def filter_csv(input_file, output_file='aavso_data.csv', convert_to_tdb=True):
    """
    Filter CSV or TXT file to keep only rows where Band='CV' and columns 'JD' and 'Magnitude'
    Optionally convert JD from UTC to TDB (Barycentric Dynamical Time)
    
    Args:
        input_file (str): Path to input CSV or TXT file
        output_file (str): Path to output CSV file (optional)
        convert_to_tdb (bool): Whether to convert JD from UTC to TDB
    """
    try:
        # Determine file extension and read accordingly
        file_extension = input_file.lower().split('.')[-1]
        
        if file_extension == 'csv':
            # Read CSV file
            df = pd.read_csv(input_file)
        elif file_extension == 'txt':
            # Try reading TXT file with different separators
            try:
                # First try comma separator
                df = pd.read_csv(input_file, sep=',')
            except:
                try:
                    # Try tab separator
                    df = pd.read_csv(input_file, sep='\t')
                except:
                    try:
                        # Try space separator
                        df = pd.read_csv(input_file, sep=' ', skipinitialspace=True)
                    except:
                        # Try semicolon separator
                        df = pd.read_csv(input_file, sep=';')
        else:
            print(f"Error: Unsupported file format '.{file_extension}'. Please use .csv or .txt files.")
            return None
        
        print(f"Successfully read {file_extension.upper()} file with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if required columns exist
        required_columns = ['Band', 'JD', 'Magnitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing columns in file: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Filter rows for appropriate band ---- CHANGE THIS FOR EACH STAR
        filtered_df = df[df['Band'] == 'CV']
        
        # Keep only JD and Magnitude columns
        result_df = filtered_df[['JD', 'Magnitude']].copy()
        
        # Remove rows with non-numeric magnitude
        result_df = remove_non_numeric_magnitude(result_df, mag_col='Magnitude')
        
        # Convert JD from UTC to TDB if requested
        if convert_to_tdb:
            print("Converting Julian Dates from UTC to Barycentric Dynamical Time (TDB)...")
            jd_tdb = jd_to_tdb(result_df['JD'].values)
            
            # Calculate the difference for information
            time_diff_seconds = (jd_tdb - result_df['JD']) * 86400
            print(f"Time correction applied: {time_diff_seconds.mean():.3f} ± {time_diff_seconds.std():.3f} seconds")
            
            # Replace JD column with TDB values
            result_df['JD'] = jd_tdb
        
        print(f"Original data: {len(df)} rows")
        print(f"Filtered data: {len(result_df)} rows (Band='CV')")
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to: {output_file}")
        
        return result_df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    import os
    
    print("AAVSO Data Filter with JD Conversion")
    print("=" * 40)
    
    # Show current directory and available CSV files
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List CSV and TXT files in current directory
    data_files = [f for f in os.listdir('.') if f.lower().endswith(('.csv', '.txt'))]
    if data_files:
        print(f"\nData files found in current directory:")
        for i, file in enumerate(data_files, 1):
            file_type = "CSV" if file.lower().endswith('.csv') else "TXT"
            print(f"  {i}. {file} ({file_type})")
        print()
    else:
        print("\nNo CSV or TXT files found in current directory.")
        print("Make sure your data file is in the same folder as this script.\n")
    
    # Prompt for input file with validation
    while True:
        input_file = input("Enter the input file name (.csv or .txt): ").strip()
        
        # Remove quotes if user copied path with quotes
        input_file = input_file.strip('"\'')
        
        if not input_file:
            print("Please enter a valid file name.")
            continue
        
        # Check file extension
        if not input_file.lower().endswith(('.csv', '.txt')):
            print("Error: Please enter a .csv or .txt file.")
            continue
            
        # Check if file exists
        if os.path.exists(input_file):
            break
        else:
            print(f"Error: File '{input_file}' not found.")
            print("Please check the file name and path.")
            
            # Suggest similar files
            if data_files:
                similar_files = [f for f in data_files if input_file.lower() in f.lower()]
                if similar_files:
                    print(f"Did you mean one of these files?")
                    for file in similar_files:
                        file_type = "CSV" if file.lower().endswith('.csv') else "TXT"
                        print(f"  - {file} ({file_type})")
            print()
    
    # Ask about TDB conversion
    while True:
        tdb_choice = input("Convert JD from UTC to Barycentric Dynamical Time (TDB)? [Y/n]: ").strip().lower()
        if tdb_choice in ['', 'y', 'yes']:
            convert_to_tdb = True
            break
        elif tdb_choice in ['n', 'no']:
            convert_to_tdb = False
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    # Output file is fixed
    output_file = 'aavso_data.csv'
    
    print(f"\nProcessing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"TDB conversion: {'Enabled' if convert_to_tdb else 'Disabled'}")
    print("-" * 40)
    
    # Filter the data file
    result = filter_csv(input_file, output_file, convert_to_tdb)
    
    if result is not None:
        print("\nFirst few rows of filtered data:")
        print(result.head())
        
        if convert_to_tdb:
            print("\nColumn descriptions:")
            print("  JD: Julian Date in Barycentric Dynamical Time (TDB)")
            print("  Magnitude: Stellar magnitude")
        else:
            print("\nColumn descriptions:")
            print("  JD: Julian Date in Universal Time (UTC)")
            print("  Magnitude: Stellar magnitude")
        
        print(f"\nProcessing complete! Results saved to '{output_file}'")
    else:
        print("\nProcessing failed. Please check the input file and try again.")

if __name__ == "__main__":
    main()
#Applying the savgol filter and plotting (This part defines the function AND removes the big features)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.interpolate import make_interp_spline as spline
import scipy.signal as signal
import matplotlib

TIC = ''

def apply_savgol_filter(time, flux, window_length_for_remove : int = 7500, mode : str = 'remove', window_length_for_gaussian : int = 100, polyorder = 4, displaygraphs : bool = True, want : str = 'df') -> pd.DataFrame:
    """
    Applies the savgol filter to `flux` and `time`.

    Works in two modes:
    - `'remove'`: Removes the outburst and other large features
    - `'gaussian'`: Finds the gaussians

    Parameters
    ----------
    time : array-like
        The time array of the lightcurve
    flux : array-like
        The flux array of the lightcurve
    mode : str
        The mode in which the function should work, either `'remove'` or `'gaussian'`
    window_length_for_gaussian : int
        The window length for the gaussian filter, works only in `'gaussian'` mode
    window_length_for_remove : int
        The window length for the remove filter, works only in `'remove'` mode
    polyorder : int
        The polynomial order for the savgol filter
    displaygraphs : bool
        (default = True)
        Whether to display the graphs or not
    want : str ('df' or 'lc')
        (default = 'df')
        Whether to return the result as a `lightkurve.LightCurve` object or a `pd.DataFrame` object

    Returns
    ----------
    if `mode == 'remove'`:
        returns the corrected lightcurve as a pandas.DataFrame object
        Columns: `'time'` and  `'flux'`
    if `mode == 'gaussian'`:
        returns the gaussians as a pandas.DataFrame object
        Columns: `'time'` and  `'gaussian_flux'`

    """
    
    if mode == 'remove':  #Removing the outburst and other large features
        flx = signal.savgol_filter(flux, int(window_length_for_remove), polyorder)
        flx2 = signal.savgol_filter(flx, int(window_length_for_remove), polyorder)

        if displaygraphs:
            #plotting the savgol
            plt.figure(figsize=(10, 5))
            plt.title('Applying SavGol to remove large features')
            plt.plot(time, flux, 'r-', lw=0.5, label='Raw Light Curve')
            plt.plot(time, flx, 'k-', lw=1, label='Initial Fit')
            plt.plot(time, flx2, 'b-', lw=1, label='Final Fit')
            plt.legend()

            #plotting the processed graph
            plt.figure(figsize=(10, 5))
            plt.title('Corrected Light Curve')
            plt.plot(time, flux - flx2, 'b-', lw=0.5)

        #Building the lightcurve again, with the correction
    
        if want == 'lc':
            l = lk.LightCurve(time = time, flux = flux - flx2)
            l.time.format = 'btjd'
            return l

        elif want == 'df':
            return pd.DataFrame({
                'time':time,
                'flux':flux - flx2
            }, index=None)

        else:
            raise ValueError("Invalid value for 'want'. Must be either 'lc' or 'df'.")


    elif mode == 'gaussian':   #Finding the gaussians, techincally
        fit_flux = signal.savgol_filter(flux, int(window_length_for_gaussian), polyorder)

        if displaygraphs:
            plt.figure(figsize=(10, 5))
            plt.title('Gaussian Fitting')
            plt.plot(time, flux, 'r-', lw=0.5, label='Corrected Light Curve')
            plt.plot(time, fit_flux, 'k-', lw=1, label='Fitted Gaussians')
            plt.legend()

        return pd.DataFrame({
            'time':time,
            'gaussian_flux': - fit_flux
        }, index=None)
    
    elif mode == 'gaussian_twice':   #Finding the gaussians, techincally
        fit_flux = signal.savgol_filter(flux, int(window_length_for_gaussian), polyorder)
        fit_fluxx = signal.savgol_filter(fit_flux, int(500), polyorder)
        if displaygraphs:
            plt.figure(figsize=(10, 5))
            plt.title('Gaussian Fitting')
            plt.plot(time, flux, 'r-', lw=0.5, label='Corrected Light Curve')
            plt.plot(time, fit_flux, 'k-', lw=1, label='Fitted Gaussians')
            plt.plot(time, fit_fluxx, 'b-', lw=1, label='Fitted Gaussians Final?')
            plt.legend()

        return pd.DataFrame({
            'time':time,
            'gaussian_flux': - fit_fluxx
        }, index=None)

#Function for checking the differences in peaks
def process_gaussians(fitted : lk.lightcurve.LightCurve | pd.DataFrame, threshold):
    """
    Finds the peaks, the periods and mean of the periods for fitted gaussians curve.

    Args:
        fitted (lightkurve.LightCurve or pd.DataFrame): 
            The lightcurve that has been fitted for Gaussians. Must have columns 'time' and 'gaussian_flux' if a DataFrame.
        threshold (float): 
            The threshold for peak detection.
        number_of_gaps (int):
            (default = 10)
            The aproximate number of gaps, always enter a much higher value than the true number.

        Returns:
        pd.DataFrame: 
            DataFrame with columns:
            - 'period': The periods between the peaks.
            - 'time': The peak times. If the period is `p2 - p1`, this returns `p2`.
    """

    import lightkurve as lk
    import pandas as pd
    from scipy.signal import find_peaks
    import numpy as np
    import matplotlib.pyplot as plt

    # Get the flux and time values
    if isinstance(fitted, lk.LightCurve):
        fl = np.array(fitted.flux)
        tm = np.array(fitted.time.btjd)
    if isinstance(fitted, pd.DataFrame):
        fl = np.array(fitted['gaussian_flux'])
        tm = np.array(fitted['time'])

    # Find peaks in fl above threshold
    peaks, _ = find_peaks(fl, height=threshold)

    # Get the associated time values for the peaks
    peak_times = tm[peaks]

    # mean_diff = np.sort(np.diff(peak_times))[: - number_of_gaps].mean()

    return pd.DataFrame({
        'period' : np.diff(peak_times),
        'time' : peak_times[1:]
    })


#Functions for rejecting outliers
def reject_outliers(data, m=1):
    removed = data[abs(data - np.mean(data)) > m * np.std(data)]
    print("Outliers Removed: ", len(removed))
    accepted = data[abs(data - np.mean(data)) < m * np.std(data)]
    plt.figure()
    plt.plot(accepted)
    return accepted

def reject_outliers_pd(data : pd.DataFrame, column_name : str, m=1):
    '''
    Removes the outliers from the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data from which the outliers are to be removed.
    column_name : str
        The column name from which the outliers are to be removed.
    m : int
        The number of standard deviations to be considered as an outlier.
    
    Returns
    --------
    accepted : pd.DataFrame
        The data with the outliers removed.
    Prints the number of outliers removed.
    
    '''
    removed = data[abs(data[column_name] - np.mean(data[column_name])) > m * np.std(data[column_name])]
    print("Outliers Removed: ", len(removed))
    accepted = data[abs(data[column_name] - np.mean(data[column_name])) < m * np.std(data[column_name])]
    accepted.reset_index(inplace=True, drop=True)
    return accepted

#Function for making the O-C diagram
def make_OC_diagram(accepted : pd.DataFrame, calculate_from : int = 1):
    '''
    Makes the O-C diagram from the accepted data.

    Parameters
    ----------
    accepted : pd.DataFrame
        The data from which the O-C diagram is to be made. Must have columns 'time' and 'period'.
    calculate_from : int
        (default = 1)
        The number of periods to calculate the CALCULATED period from.
    
    Returns
    --------
    OC_DataFrame : pd.DataFrame
        The O-C diagram data.
        Columns:
        - 'T' : Time of the period from the start
        - 'E' : Event Number
        - 'O-C' : The O-C values
    '''
    df = accepted.copy(deep=True)

    df['T'] = df['time'] - df['time'][0]
    p0 = df['period'][:calculate_from].mean()
    df['p_'] = [ ( df['period'][:x].sum() / x ) for x in range(1, len(df) + 1)]
    df['E'] = df['T'] / df['p_']
    df['E'] = df['E'].astype(int)
    df['O-C'] = df['T'] * ( 1 - ( p0 / df['p_'] ) )
    return df[['T', 'E', 'O-C']]


def straight_lines(lightcurve : lk.lightcurve.LightCurve, cadence_magnifier : int = 4) -> lk.lightcurve.LightCurve:
    '''
    Takes in a lightcurve and fills the gaps with a straight line, furthermore, smoothens the lightcurve with a spline interpolation with a factor of `cadence_magnifier`.
    Returns a lightcurve
    '''

    #BASICS
    lc = pd.DataFrame({'time': lightcurve.time.jd, 'flux': np.array(lightcurve.flux, dtype='d')})
    lc.dropna(inplace=True)
    cadence_in_days = ((np.median(np.diff(lc['time'][:100])) * 86400).round())/86400
    flux = np.array(lightcurve.flux, dtype='d')
    time = np.array(lightcurve.time.jd)

    #PEAKS
    peaks, _ = signal.find_peaks(np.diff(time), height = cadence_in_days * 10)
    print(f"Gaps at times: {time[peaks] - 2457000}")

    #Filling the Gaps
    for i in peaks:
        t = (time[i], time[i+1])
        f = (flux[i], flux[i+1])
        df = pd.DataFrame({'time': t, 'flux': f})
        df.set_index('time', inplace = True)
        df = df.reindex(np.arange(time[i], time[i+1], cadence_in_days))
        df['flux'] = df['flux'].interpolate('linear')
        df.reset_index(inplace = True)
        df.rename(columns={'index':'time'}, inplace=True)
        lc = pd.concat([df[1:-1], lc]).sort_values('time')

    time_final = np.array(lc['time'])
    flux_final = np.array(lc['flux'])
    time_smooth = np.linspace(time_final.min(), time_final.max(), len(time_final) * cadence_magnifier)
    flux_smooth = spline(time_final, flux_final, k = 3)(time_smooth)

    # lightcurve.flux, lightcurve.time = flux_smooth, time_smooth

    disposable_lightcurve = lk.LightCurve(time = time_smooth, flux = flux_smooth)
    disposable_lightcurve.time.format = 'btjd' 

    return disposable_lightcurve

def spline_while_jumping_gaps(lightcurve : lk.lightcurve.LightCurve, cadence_magnifier : int = 4) -> lk.lightcurve.LightCurve:
    """Takes in a lightcurve and smoothens the lightcurve with a spline interpolation with a factor of `cadence_magnifier`.
    Returns a lightcurve"""

    #BASICS
    lc = pd.DataFrame({'time': lightcurve.time.jd, 'flux': np.array(lightcurve.flux, dtype='d')})
    lc.dropna(inplace=True)
    cadence_in_days = ((np.median(np.diff(lc['time'][:100])) * 86400).round())/86400
    flux = np.array(lightcurve.flux, dtype='d')
    time = np.array(lightcurve.time.jd)

    #PEAKS
    peaks, _ = signal.find_peaks(np.diff(time), height = cadence_in_days * 10)
    print(f"Gaps at times: {time[peaks] - 2457000}")

    lightcurve_df = pd.DataFrame({'time':[], 'flux':[]})

    peaks = np.append(peaks, len(time) - 1)
    current_begin = 0

    for peak in peaks:
        current_end = peak
        time_smooth = np.linspace(time[current_begin], time[current_end], int(((time[current_end] - time[current_begin]) / cadence_in_days ) * cadence_magnifier))
        flux_smooth = spline(time[current_begin:current_end], flux[current_begin:current_end], k = 3)(time_smooth)
        lightcurve_df = pd.concat([lightcurve_df, pd.DataFrame({'time':time_smooth, 'flux':flux_smooth})]).sort_values('time')
        current_begin = peak + 1

    disposable_lightcurve = lk.LightCurve(time = lightcurve_df['time'], flux = lightcurve_df['flux'])
    disposable_lightcurve.time.format = 'btjd' 

    return disposable_lightcurve


def get_lightcurves(TIC, use_till = 30, use_from = 0, author = None, cadence = None) -> list:
    """
    Returns lightcurves for a set TIC from the TESS database.
    
    Parameters
    ----------
    TIC : int
        The TIC number of the system.
    use_till : int
        (default = 30)
        The number of lightcurves to be used.
    use_from : int
        (default = 0)
        The number of lightcurves to be skipped.
    author : str
        (default = None)
        The authors for the lightcurve. Eg. 'SPOC', 'QLP', etc.
    cadence : str or float
        (default = None)
        The cadence of the lightcurve, used interchangably with exptime. Eg. 'long', 'short', float, etc.

    Returns
    --------
    lcs : list
        The list of lightcurves.
    """
    search_results = lk.search_lightcurve(TIC, author = author, cadence = cadence)
    print(search_results[use_from:use_till])
    sorted_search_results = sorted(search_results[use_from:use_till], key=lambda x: x.mission)

    lcs = []

    for s in sorted_search_results:
        try:
            lcs.append(s.download())
        except:
            pass

    return lcs


def combine_lightcurves(lcs : list[lk.lightcurve.LightCurve]) -> lk.lightcurve.LightCurve:
    """
    Combines multiple lightcurves into one.

    Parameters
    ----------
    lcs : list[lightkurve.LightCurve]
        List of lightcurves to be combined.
    
    Returns
    -------
    lightcurve : lightkurve.LightCurve
        Combined lightcurve.
    """
    lightcurve_df = pd.DataFrame({
        'time' : np.concatenate([lc.time.jd for lc in lcs]),
        'flux' : np.concatenate([lc.flux for lc in lcs])
    })
    lightcurve_df.sort_values('time', inplace=True)
    lightcurve_df.info()
    #print(lightcurve_df.memory_usage())

    lc = lk.LightCurve(time= lightcurve_df['time'], flux= lightcurve_df['flux'])
    lc.time.format = 'btjd'
    return lc

def gaussian(x, amp, cen, wid, inverse : bool = False):
    """
    Returns a gaussian function.
    
    Parameters
    ----------
    x : array_like or float
        The x value(s) for which the gaussian is to be calculated.
    amp : float
        The amplitude of the gaussian.
    cen : float
        The center of the gaussian.
    wid : float
        The width of the gaussian.
    
    Returns
    -------
    y : array_like or float
        The value of the gaussian at `x`.
    """
    if inverse == True:
        return - (amp * np.exp(-(x - cen)**2 / (2 * wid**2)))
    else:
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def sine(x, amp, freq, phase, offset):
    """
    Returns a sine function.
    
    Parameters
    ----------
    x : array_like or float
        The x value(s) for which the sine is to be calculated.
    amp : float
        The amplitude of the sine.
    freq : float
        The frequency of the sine.
    phase : float
        The phase of the sine.
    offset : float
        The offset of the sine.
    
    Returns
    -------
    y : array_like or float
        The value of the sine at `x`.
    """
    return amp * np.sin(2 * np.pi * freq * (x - phase)) + offset
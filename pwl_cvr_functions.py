import numpy as np
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
import copy
import csv

def get_endtidal_peaks(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3):
    '''peaks_inds, peaks_vals, peaks_df = get_endtidal_peaks(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for peaks (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for peaks (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    peak (default 0), and prominence is the minimum prominence of the peak (default 3).'''
    # Get the end tidal peaks from the data
    peaks_inds, peaks_vals = find_peaks(data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift CO2peaks_inds by the starting index of df_baseline1
    peaks_inds = peaks_inds + data_init_index
    # Create pandas dataframe of the peaks
    peaks_vals = peaks_vals['peak_heights']
    peaks_df = pd.DataFrame([{'index': peaks_inds[i], 'val': peaks_vals[i]} for i in range(len(peaks_vals))])
    return peaks_inds, peaks_vals, peaks_df

def get_endtidal_valleys(data, data_init_index, search_window=6, sample_rate=200, height=-20, prominence=2):
    '''valleys_inds, valleys_vals, valleys_df = get_endtidal_valleys(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for valleys (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for valleys (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    valley (default -20), and prominence is the minimum prominence of the valley (default 2).'''
    # Get the end tidal valleys from the data
    valleys_inds, valleys_vals = find_peaks(-data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift valleys_inds by the starting index of df_baseline1
    valleys_inds = valleys_inds + data_init_index
    # Create pandas dataframe of the valleys
    valleys_vals = -valleys_vals['peak_heights']
    valleys_df = pd.DataFrame([{'index': valleys_inds[i], 'val': valleys_vals[i]} for i in range(len(valleys_vals))])
    return valleys_inds, valleys_vals, valleys_df

def get_average_breathing_rate(data, sample_rate):
    '''br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    X = np.fft.fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to 0.05 Hz
    idx = (np.abs(f_oneside - 0.05)).argmin()
    # Find the index of the max amplitude above 0.05 Hz 
    idx_max = idx + (np.abs(X[idx:n_oneside])).argmax()
    br_avg = f_oneside[idx_max]
    return br_avg, f_oneside, X, n_oneside

def plot_breathing_rate_fft(data, sample_rate):
    '''plot_breathing_rate_fft(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    print('The average baseline breathing rate is:', br_avg, 'Hz','1/br_avg:', 1/br_avg, 's')
    print('Breaths per minute:', (br_avg*60), 'BPM')

    plt.figure(figsize = (6, 3))
    plt.xlim(0, 0.5)
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title('Breathing Rate FFT Spectrum')
    plt.show()

def get_average_tcd_rate(data, sample_rate):
    '''br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    X = np.fft.fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to 0.4 Hz
    idx = (np.abs(f_oneside - 0.4)).argmin()
    # Find the index of the max amplitude above 0.4 Hz 
    idx_max = idx + (np.abs(X[idx:n_oneside])).argmax()
    tcd_avg = f_oneside[idx_max]
    return tcd_avg, f_oneside, X, n_oneside

def plot_tcd_fft(data, sample_rate):
    '''plot_breathing_rate_fft(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    tcd_avg, f_oneside, X, n_oneside = get_average_tcd_rate(data, sample_rate)
    print('The average tcd rate is:', tcd_avg, 'Hz', 1/tcd_avg, 's')
    print('Pulses per minute:', (tcd_avg*60), 'BPM')

    plt.figure(figsize = (6, 3))
    plt.xlim(0, 4)
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title('TCD Rate FFT Spectrum')
    plt.show()

def get_data_between_comments(dataframe, comment1_index, comment2_index, crop_start, crop_end):
    '''df_5co2 = get_data_between_comments(dataframe, comment1, comment2, crop_start, crop_end):
    where dataframe is the dataframe to be analyzed, comment1 is the first comment to start the new dataframe, and comment2 is the comment index to end the new dataframe.'''
    sample_rate = 1/(float(dataframe.iloc[2][0]) - float(dataframe.iloc[1][0]))
    # Get the rows of the comments
    rows_comments = dataframe[dataframe['Comments'].notnull()].index
    # Crop the DataFrame to the rows after comment1_index and before comment2_index
    df_between_comments = dataframe.iloc[rows_comments[comment1_index]:rows_comments[comment2_index]]
    return df_between_comments.iloc[
        (int(crop_start * sample_rate)) : -(int(crop_end * sample_rate))
    ]

def get_data_window_between_comments(dataframe, comment1_index, comment2_index, crop_start_seconds, window_length_seconds):
    '''df_5co2 = get_data_between_comments(dataframe, comment1, comment2, crop_start_seconds, window_length_seconds)
    where dataframe is the dataframe to be analyzed, comment1 is the first comment to start the new dataframe, and comment2 is the comment index to end the new dataframe.'''
    sample_rate = 1/(float(dataframe.iloc[2][0]) - float(dataframe.iloc[1][0]))
    # Get the rows of the comments
    rows_comments = dataframe[dataframe['Comments'].notnull()].index
    return dataframe.iloc[
        (rows_comments[comment1_index]
        + (int(crop_start_seconds * sample_rate))) : (
            (rows_comments[comment1_index] + int((crop_start_seconds + window_length_seconds) * sample_rate))
        )
    ]



def get_tcd_peaks(data, data_init_index, search_window=0.5, sample_rate=200, height=0, prominence=15):
    '''peaks_inds, peaks_vals, peaks_df = get_tcd_peaks(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for peaks (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for peaks (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    peak (default 0), and prominence is the minimum prominence of the peak (default 15).'''
    # Get the end tidal peaks from the data
    peaks_inds, peaks_vals = find_peaks(data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift CO2peaks_inds by the starting index of df_baseline1
    peaks_inds = peaks_inds + data_init_index
    # Create pandas dataframe of the peaks
    peaks_vals = peaks_vals['peak_heights']
    peaks_df = pd.DataFrame([{'index': peaks_inds[i], 'val': peaks_vals[i]} for i in range(len(peaks_vals))])
    return peaks_inds, peaks_vals, peaks_df

def get_tcd_valleys(data, data_init_index, search_window=0.5, sample_rate=200, height=-100, prominence=15):
    '''valleys_inds, valleys_vals, valleys_df = get_tcd_valleys(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for valleys (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for valleys (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    valley (default -100), and prominence is the minimum prominence of the valley (default 15).'''
    # Get the end tidal valleys from the data
    valleys_inds, valleys_vals = find_peaks(-data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift valleys_inds by the starting index of df_baseline1
    valleys_inds = valleys_inds + data_init_index
    # Create pandas dataframe of the valleys
    valleys_vals = -valleys_vals['peak_heights']
    valleys_df = pd.DataFrame([{'index': valleys_inds[i], 'val': valleys_vals[i]} for i in range(len(valleys_vals))])
    return valleys_inds, valleys_vals, valleys_df

def high_pass_filter(data, sample_rate, cutoff_freq):
    '''filtered_data = high_pass_filter(data, sample_rate, cutoff_freq)
    where data is the data to be filtered (pandas Dataframe), sample_rate is the sample rate of the data (Hz), and cutoff_freq is the cutoff frequency (Hz).'''
    # Get the FFT of the data
    X = np.fft.fft(data)
    # Get the number of data points
    N = len(X)
    # Get the frequency
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to cutoff_freq
    idx = (np.abs(f_oneside - cutoff_freq)).argmin()
    # Filter the data
    X[idx:n_oneside] = 0
    # Get the filtered data
    filtered_data = np.real(np.fft.ifft(X))
    # replace values in dataframe with filtered data
    filtered_df = data
    filtered_df[:] = filtered_data
    return filtered_df

def low_pass_filter(data, sample_rate, cutoff_freq):
    '''filtered_data = low_pass_filter(data, sample_rate, cutoff_freq)
    where data is the data to be filtered (pandas Dataframe), sample_rate is the sample rate of the data (Hz), and cutoff_freq is the cutoff frequency (Hz).'''
    # Get the FFT of the data
    X = np.fft.fft(data)
    # Get the number of data points
    N = len(X)
    # Get the frequency
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to cutoff_freq
    idx = (np.abs(f_oneside - cutoff_freq)).argmin()
    # Filter the data
    X[idx:] = 0
    # Get the filtered data
    filtered_data = np.real(np.fft.ifft(X))
    # replace values in dataframe with filtered data
    filtered_df = copy.deepcopy(data)
    filtered_df[:] = filtered_data
    return filtered_df

def align_peaks_and_valleys(peaks_inds, peaks_vals, valleys_inds, valleys_vals, TCD_data):
    '''MCAvmean_inds, MCAvmean_vals, MCAvmean_df = align_peaks_and_valleys(peaks_inds, peaks_vals, valleys_inds, valleys_vals, TCD_data)
    where peaks_inds is the indices of the peaks (pandas Dataframe), peaks_vals is the values of the peaks (pandas Dataframe), valleys_inds is the indices of the valleys (pandas Dataframe), valleys_vals is the values of the valleys (pandas Dataframe), and TCD_data is the TCD data (pandas Dataframe).'''
    
    #drop to last peak or trough if necessary to make the number of peaks and troughs the same
    if len(valleys_vals)>len(peaks_vals):
        valleys_vals = valleys_vals[:len(peaks_vals)]
        valleys_inds = valleys_inds[:len(peaks_inds)]
    if len(valleys_vals)<len(peaks_vals):
        peaks_vals = peaks_vals[:len(valleys_vals)]
        peaks_inds = peaks_inds[:len(valleys_inds)]

    #find the mean of the peaks and troughs
    # MCAvmean_inds = (
    #     (peaks_inds - valleys_inds) + peaks_inds
    #     if peaks_inds[0] > valleys_inds[0]
    #     else (peaks_inds - valleys_inds) + valleys_inds
    # )
    # MCAvmean_vals = [
    #     TCD_data[MCAvmean_inds[i]] for i in range(len(MCAvmean_inds))
    # ]
    # MCAvmean_df = pd.DataFrame([{'index': MCAvmean_inds[i], 'val': MCAvmean_vals[i]} for i in range(len(MCAvmean_vals))])

    if peaks_inds[0]>valleys_inds[0]:
        MCAvmean_inds = (peaks_inds-valleys_inds)+peaks_inds
    else:
        MCAvmean_inds = (peaks_inds-valleys_inds)+valleys_inds

        MCAvmean_vals = ((peaks_vals-valleys_vals)/2)+valleys_vals

    MCAvmean_vals = ((peaks_vals-valleys_vals)/2)+valleys_vals
    MCAvmean_df = pd.DataFrame([{'index': MCAvmean_inds[i], 'val': MCAvmean_vals[i]} for i in range(len(MCAvmean_vals))])


    return MCAvmean_inds, MCAvmean_vals, MCAvmean_df

def get_df(filepath):
    '''df = get_df(filepath)
    where filepath is the path to the text file to be read with 5 columns: O2, CO2, TCD, Pulse, Comments, and each row is a time point.'''
    
    # Read text file into pandas DataFrame
    cols = pd.read_csv(filepath, sep='\t', nrows=5, header=(4)).columns
    # Delete the first column
    cols = cols.drop(cols[0])
    cols = cols.insert(0, 'Time')
    # Add a column to the DataFrame
    cols = cols.insert(5, 'Comments')
    # Read the file into a DataFrame
    df = pd.read_csv(filepath, sep='\t',skiprows=(0,1,2,3,4,5),header=None, names=cols)

    return df

def get_segment_lengths(filepath, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2):
    '''len_base1, len_5co2, len_base2 = get_segment_lengths(filepath, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2)
    expecting 4 channels of data: O2, CO2, TCD, and Pulse'''
    
    df = get_df(filepath)
 
    # List the rows with comments
    rows_comments=df[df['Comments'].notnull()]  

    # Get the length of each breathing segment
    df_baseline1 = get_data_between_comments(df, comment1_index=comment_start_baseline1, comment2_index=comment_end_baseline1, crop_start=0, crop_end=1)
    df_5co2 = get_data_between_comments(df, comment1_index=comment_start_5co2, comment2_index=comment_end_5co2, crop_start=0, crop_end=1)
    df_baseline2 = get_data_between_comments(df, comment1_index=comment_start_baseline2, comment2_index=comment_end_baseline2, crop_start=0, crop_end=1)

    len_base1 = len(df_baseline1)/200
    len_5co2 = len(df_5co2)/200
    len_base2 = len(df_baseline2)/200
    
    print('Baseline 1: ', len_base1, 's ||','5% CO2: ', len_5co2, 's ||','Baseline 2: ', len_base2, 's')
    return len_base1, len_5co2, len_base2

def separate_data(df, TCD_calibration_factor, TCD_threshold):
    '''TCD_th, CO2, O2, PPG = separate_data(df, TCD_calibration_factor, TCD_threshold)'''

    # For baseline1 get raw TCD data
    TCD = df.iloc[:,3].astype(float)*TCD_calibration_factor
    # For baseline 1 get raw CO2 and O2 data
    CO2 = df.iloc[:,1].astype(float)
    O2 = df.iloc[:,2].astype(float)
    # For baseline 1 get raw PPG data
    PPG = df.iloc[:,4].astype(float)

    # Threshold TCD
    # Replace values in TCD_baseline1 below threshold with NaN
    TCD_th = copy.deepcopy(TCD)
    TCD_th[TCD_th < TCD_threshold] = np.nan

    return TCD_th, CO2, O2, PPG

def get_cvr(filepath, TF_to_plot, save_fig, save_csv, csv_filename, save_log, log_filename, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2, TCD_calibration_factor, TCD_threshold, CO2prominence_base, O2prominence_base, CO2prominence_5co2, O2prominence_5co2, P_conversion_perc2mmHg, window_length_baseline1, window_length_5co2, window_length_baseline2):
    '''percent_CVR, CVR, CVR_SI = cvr_func.get_cvr(filepath, TF_to_plot, save_fig, save_csv, csv_filename, save_log, log_filename, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2, TCD_calibration_factor, TCD_threshold, CO2prominence_base, O2prominence_base, CO2prominence_5co2, O2prominence_5co2, P_conversion_perc2mmHg, window_length_baseline1, window_length_5co2, window_length_baseline2)'''
    print('---------------------')
    df = get_df(filepath)

    len_base1, len_5co2, len_base2 = get_segment_lengths(filepath, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2)

    df_baseline1 = get_data_window_between_comments(df, comment1_index=comment_start_baseline1, comment2_index=comment_end_baseline1, crop_start_seconds=30, window_length_seconds=window_length_baseline1)
    df_5co2 = get_data_window_between_comments(df, comment1_index=comment_start_5co2, comment2_index=comment_end_5co2, crop_start_seconds=30, window_length_seconds=window_length_5co2)
    df_baseline2 = get_data_window_between_comments(df, comment1_index=comment_start_baseline2, comment2_index=comment_end_baseline2, crop_start_seconds=20, window_length_seconds=window_length_baseline2)
    print('-------After Cropping-------')
    print('Baseline 1: ', float(df_baseline1.iloc[-1][0]) - float(df_baseline1.iloc[0][0]), 's ||','5% CO2: ', float(df_5co2.iloc[-1][0]) - float(df_5co2.iloc[0][0]), 's ||','Baseline 2: ', float(df_baseline2.iloc[-1][0]) - float(df_baseline2.iloc[0][0]), 's')

    # Get the average breathing rate
    # Determine average breathing rate during baseline to determine the end-tidal search window using a fourier transform
    baseline_co2_data = df_baseline1.iloc[:,1].astype(float)
    sample_rate = 1/(float(df_5co2.iloc[2][0]) - float(df_5co2.iloc[1][0]))

    br_avg, _, _, _ = get_average_breathing_rate(baseline_co2_data, sample_rate)

    TCD_baseline1_th, CO2_baseline1, O2_baseline1, PPG_baseline1 = separate_data(df_baseline1, TCD_calibration_factor, TCD_threshold)
    TCD_5co2_th, CO2_5co2, O2_5co2, PPG_5co2 = separate_data(df_5co2, TCD_calibration_factor, TCD_threshold)

    # Define search window as 0.6 seconds longer than the average breathing rate
    search_window = (1/br_avg)*0.6 #- 3.5 # seconds
    print('Sample rate:', sample_rate, 'Hz ||', 'Search window:', search_window, 'seconds')
    # Get CO2 end tidal peaks from the baseline data
    CO2peaks_base1_inds, CO2peaks_base1_vals, CO2peaks_base1_df = get_endtidal_peaks(np.array(CO2_baseline1), CO2_baseline1.index[0], search_window=search_window, sample_rate=200, height=0, prominence=CO2prominence_base)
    # Get the O2 end tidal valleys from the baseline data
    O2valleys_base1_inds, O2valleys_base1_vals, O2valleys_base1_df = get_endtidal_valleys(np.array(O2_baseline1), O2_baseline1.index[0], search_window=search_window, sample_rate=200, height=-30, prominence=O2prominence_base)

    # Get the mean of the thresholded TCD data
    MCAvmean_base = np.mean(TCD_baseline1_th)
    # Get the mean values of the CO2 and O2 end tidal peaks
    CO2_mean_base = np.mean(CO2peaks_base1_vals)
    O2_mean_base = np.mean(O2valleys_base1_vals)

    # Get CO2 end tidal peaks from the baseline data
    CO2peaks_5co2_inds, CO2peaks_5co2_vals, CO2peaks_5co2_df = get_endtidal_peaks(np.array(CO2_5co2), CO2_5co2.index[0], search_window=search_window, sample_rate=200, height=0, prominence=CO2prominence_5co2)
    # Get the O2 end tidal valleys from the baseline data
    O2valleys_5co2_inds, O2valleys_5co2_vals, O2valleys_5co2_df = get_endtidal_valleys(np.array(O2_5co2), O2_5co2.index[0], search_window=search_window, sample_rate=200, height=-20, prominence=O2prominence_5co2)

    # Get the mean of the thresholded TCD data, CO2 and O2 end tidal peaks
    MCAvmean_5co2 = np.mean(TCD_5co2_th)
    CO2_mean_5co2 = np.mean(CO2peaks_5co2_vals)
    O2_mean_5co2 = np.mean(O2valleys_5co2_vals)
    
    print('---------------------')
    print('--- BASELINE DATA ---')
    print('end tidal CO2 mean:', CO2_mean_base*P_conversion_perc2mmHg, 'mmHg')
    print('end tidal O2 mean:', O2_mean_base*P_conversion_perc2mmHg, 'mmHg')
    print('Stimulus index:', CO2_mean_base/O2_mean_base)
    print('MCAvmean mean_base:', MCAvmean_base, 'cm s-1')
    print('--- 5% CO2 DATA ---')
    print('end tidal CO2 mean:', CO2_mean_5co2*P_conversion_perc2mmHg, 'mmHg')
    print('end tidal O2 mean:', O2_mean_5co2*P_conversion_perc2mmHg, 'mmHg')
    print('Stimulus index:', CO2_mean_5co2/O2_mean_5co2)
    print('MCAvmean mean_5CO2:', MCAvmean_5co2, 'cm s-1')
    print('---------------------')

    if TF_to_plot==True:
        plot_4channel_data('Baseline 1', filepath, TCD_baseline1_th, CO2_baseline1, CO2peaks_base1_df, O2_baseline1, O2valleys_base1_df, PPG_baseline1, save_fig, MCAvmean=MCAvmean_base)
        plot_4channel_data('5% CO2', filepath, TCD_5co2_th, CO2_5co2, CO2peaks_5co2_df, O2_5co2, O2valleys_5co2_df, PPG_5co2, save_fig, MCAvmean=MCAvmean_5co2)
    else:
        print('TF_to_plot is False')

    percent_cvr = 100*((MCAvmean_5co2-MCAvmean_base)/MCAvmean_base)
    cvr = 100*((MCAvmean_5co2-MCAvmean_base)/MCAvmean_base)/(CO2_mean_5co2*P_conversion_perc2mmHg-CO2_mean_base*P_conversion_perc2mmHg)
    cvr_si = 100*((MCAvmean_5co2-MCAvmean_base)/MCAvmean_base)/((CO2_mean_5co2/O2_mean_5co2)*P_conversion_perc2mmHg-(CO2_mean_base/O2_mean_base)*P_conversion_perc2mmHg)

    # Save results to CSV file
    filename = os.path.basename(filepath)
    csv_filename = 'pwl_outputs/'+ csv_filename
    if save_csv == True:

        if os.path.exists(csv_filename):
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, CO2prominence_base, O2prominence_base, CO2prominence_5co2, O2prominence_5co2, MCAvmean_base, MCAvmean_5co2, CO2_mean_base, CO2_mean_5co2, O2_mean_base, O2_mean_5co2, percent_cvr, cvr, cvr_si])
        else:
            # Write the header and the data
            with open(csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Input filename", "CO2prominence_base", "O2prominence_base", "CO2prominence_5co2", "O2prominence_5co2", "MCAvmean_base", "MCAvmean_5co2", "ETCO2_mean_base", "ETCO2_mean_5co2", "ETO2_mean_base", "ETO2_mean_5co2", "Percent_CVR", "CVR", "CVR_SI"])
                writer.writerow([filename, CO2prominence_base, O2prominence_base, CO2prominence_5co2, O2prominence_5co2, MCAvmean_base, MCAvmean_5co2, CO2_mean_base, CO2_mean_5co2, O2_mean_base, O2_mean_5co2, percent_cvr, cvr, cvr_si])

    # Save summary of data timeseries to CSV file
    csv_summary_filename = 'pwl_outputs/'+ log_filename
    if save_log == True:
        if os.path.exists(csv_summary_filename):
            with open(csv_summary_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, len_base1, len_5co2, len_base2, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2])
                
        else:
            # Write the header and the data
            with open(csv_summary_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Input filename", "Length of baseline1 (s)", "Length of 5 CO2 (s)", "Length of baseline2 (s)", "comment_start_baseline1", "comment_end_baseline1", "comment_start_5co2", "comment_end_5co2", "comment_start_baseline2", "comment_end_baseline2"])
                writer.writerow([filename, len_base1, len_5co2, len_base2, comment_start_baseline1, comment_end_baseline1, comment_start_5co2, comment_end_5co2, comment_start_baseline2, comment_end_baseline2])
    
    return percent_cvr, cvr, cvr_si

def plot_4channel_data(plot_title, filepath, TCD_th, CO2, CO2_peaks, O2, O2_valleys, PPG, save_fig, MCAvmean):
    '''plot_4channel_data(plot_title, filepath, TCD_th, CO2, CO2_peaks, O2, O2_valleys, PPG, save_fig, MCAvmean)'''
    
    #plot
    fig = plt.figure()

    #plot parameters
    fig.set_figheight(18)
    fig.set_figwidth(18)
    tfont = {'fontname':'Times'}

    #plot TCD MCAv
    ax1 = fig.add_subplot(411)
    ax1.set_title(plot_title, fontsize=18, **tfont)       
    ax1.set_ylabel('Blood Velocity (cm/s)', fontsize=18, **tfont)
    TCD_th.plot(c="black", lw=2)
    ax1.axhline(y=MCAvmean, color='r', linestyle='-', label='MCAvmean')
    ax1.legend(['MCAvmax', 'MCAvmin', 'MCAvmean'], fontsize=10)

    #plot CO2 percentage
    ax2 = fig.add_subplot(412)   
    ax2.set_ylabel('CO$_{2}$ (%)', fontsize=18, **tfont)
    CO2.plot(c="blue", lw=2)
    CO2_peaks.plot(ax=ax2,x='index', y = 'val', marker='*',linestyle='None',markersize = 20.0, color = 'r')
    ax2.legend(['End-Tidal CO2'], fontsize=10)


    #plot O2 percentage
    ax3 = fig.add_subplot(413)     
    ax3.set_ylabel('O$_{2}$ (%)', fontsize=18, **tfont)
    O2.plot(c="red", lw=2)
    O2_valleys.plot(ax=ax3,x='index', y = 'val', marker='*',linestyle='None',markersize = 20.0, color = 'g')
    ax3.legend(['End-Tidal O2'], fontsize=10)


    #plot PPG readout(V)
    ax4 = fig.add_subplot(414)   
    ax4.set_xlabel('Time (ms)', fontsize=18, **tfont)    
    ax4.set_ylabel('PPG (V)', fontsize=18, **tfont)
    PPG.plot(c="black", lw=2)

    plt.show()

    if save_fig == True:
        # Create folder if it doesn't exist
        if not os.path.exists('pwl_outputs/PWL_figures'):
            os.makedirs('pwl_outputs/PWL_figures')
        # Save the figure in the folder
        filename = os.path.basename(filepath)
        # remove ending from filename
        fig.savefig('pwl_outputs/PWL_figures/' + filename[:-4] + '_' + plot_title + '.png')

    return

def reorder_csv_by_subj(csv_filename):
    # Reorder csv rows by subject number in filename (ascending)
    df = pd.read_csv(csv_filename)
    df['Subject'] = df['Input filename'].str.extract('(\d+)', expand=False).astype(int)
    df = df.sort_values(by=['Subject'])
    df.to_csv(csv_filename, index=False)
    return

def create_consolidated_csv(ses01_csv_filename, ses02_csv_filename, output_filename):
    # Create a consolidated csv file with all the data from ses02 appended with all but the first 5 rows of ses01
    df_ses01 = pd.read_csv(ses01_csv_filename)
    df_ses02 = pd.read_csv(ses02_csv_filename)
    #df_ses02 = df_ses02.append(df_ses02, ignore_index = True)
    df_out = pd.concat([df_ses02, df_ses01.iloc[4:]], ignore_index=True)
    #df_ses02.loc[len(df_ses02.index)] = df_ses01.iloc[5:]
    df_out.to_csv(output_filename, index=False)

    return

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lmfit as lm
import scipy
from scipy.optimize import curve_fit

def obtain_data(filename):
    # reading in the csv
    df = pd.read_csv(filename)

    # print(df.head())
    # converting the strings from the csv into numbers
    df['x-axis'] = pd.to_numeric(df['x-axis'], errors='coerce')
    df['1'] = pd.to_numeric(df['1'], errors='coerce')
    df['2'] = pd.to_numeric(df['2'], errors='coerce')
    # dropping the rows that have NaN
    df = df.dropna(subset=['x-axis', '1', '2'])
    print(df.head())


    plt.figure()
    plt.plot(df['x-axis'], df['1'])
    plt.plot(df['x-axis'], df['2'])
    plt.xlabel('time (s)')
    plt.ylabel('voltage')
    # plt.plot(df['x-axis'], df['1'])

    return df


#################################################################
# I will now include various methods for finding the spacing b/w the peaks
# they should move from less complex to more complex

#################################################################
# using the scipy peak finding algorithm to locate the peaks
def peak_finding(df, peak_width, channel):
    peaks = scipy.signal.find_peaks(df[channel], width=peak_width)
    # print(peaks[1])
    plt.figure()
    plt.plot(df['x-axis'], df[channel])
    plt.scatter(df['x-axis'][peaks[0] + 8], peaks[1]['prominences'] + 0.36, c = 'm', s=15)
    # the +8 is to account for the NaN indices which were removed
    # the +36 is a y-axis normalization
    plt.xlabel('time (s)')
    plt.ylabel('voltage')

    return(peaks)


#################################################################
# now i'll take the two middle peaks and calculate delta t with them
def delta_t_calc(df, peaks):
    mid_point = int(len(peaks[0])/2)
    time_diff = df['x-axis'][peaks[0][mid_point]] - df['x-axis'][peaks[0][mid_point - 1]]
    print(f'the time difference is {time_diff:.4f} seconds')
    return time_diff

#################################################################
# instead i can plot the times vs frequency and take the linear part of the polynomial
# for the frequencies, I will note that they are integer steps in the free spectral range
def lin_fitting(x_vals: list, y_vals: list):

    # define the Model
    l_model = lm.models.LinearModel(independent_vars='x')

    # create parameters
    my_params = l_model.make_params()

    # perform fit
    fit_result = l_model.fit(y_vals, x = x_vals, params = my_params)

    # acquiring data from the fit
    l_vals = fit_result.eval(x=x_vals)

    intercept = fit_result.params['intercept'].value
    slope = fit_result.params['slope'].stderr

    print(f'The equation of the line is y={slope:.4f}x + {intercept:.4f}')

    print(fit_result.fit_report())
    
    return (l_vals, slope, intercept)

def pol_fitting(x_vals: list, y_vals: list):
    def func(x, a, b, c, d):
        return a + b * x + c * x ** 2 + d * x ** 3
    
    popt, _ = curve_fit(func, x_vals, y_vals)
    print(f'polynomial fit result: {popt}')
    # print(popt)
    x_fit = np.linspace(min(x_vals), max(x_vals), 500)
    y_fit = func(x_fit, *popt)
    return x_fit, y_fit


def delta_t_fit(df, peaks, fsr):
    plt.figure()
    y_ax = np.arange(len(peaks[0]))
    x_ax = df['x-axis'].iloc[peaks[0]]

    x_ax = x_ax[1:-1]
    y_ax = y_ax[1:-1]*fsr
    # plt.scatter(df['x-axis'][peaks[0][2:-1]], peaks[1]['prominences'][2:-1] + 0.36, s=15)
    plt.scatter(x_ax, y_ax)

    lin_fit = lin_fitting(x_ax[3:-3], y_ax[3:-3]) # to take the middle sections of the line
    plt.plot(x_ax[3:-3], lin_fit[0], color='darkgreen', label = 'linear fit')
    # lin_fit = lin_fitting(df['x-axis'][peaks[0][5:-4]], y_ax[3:-3])
    # plt.plot(df['x-axis'][peaks[0][5:-4]], lin_fit[0], color='darkgreen', label = 'linear fit')
    
    x_data, y_data = pol_fitting(x_ax, y_ax)
    plt.plot(x_data, y_data, color='orange', label = 'polynomial fit')
    plt.legend()
    plt.ylabel('frequency')
    plt.xlabel('time')

    time_diff = lin_fit[1] # 1/(the slope)
    return time_diff

def fsr_calc(l, n):
    # need index of refraction of silicon at 1580nm
    # USE THIS CALCULATED FSR TO CALIBRATE THE Y-AXIS OF THE CALIBRATION PLOT
    fsr = (scipy.constants.c / (2 * l * n) ) * 1E-6 # reported in MHz
    print(f'the free spectral range is {fsr}')
    return fsr


def main():
    # defining necessary constants, etc.
    channel = '1'
    length = 16.483 * 10 # cm
    n = 3.45 # refractive index of Silicon https://srd.nist.gov/jpcrdreprint/1.555624.pdf

    # analyzing the data
    data_df = obtain_data('scope_5.csv')
    data2_df = obtain_data('scope_6.csv')
    peaks = peak_finding(data_df, 10, channel)
    time_diff1 = delta_t_calc(data_df, peaks)
    fsr = fsr_calc(length, n)
    time_diff2 = delta_t_fit(data_df, peaks, fsr)
    print(f'time difference with no fitting: {time_diff1}')
    print(f'time difference with linear fitting: {time_diff2}')
    plt.show()


main()

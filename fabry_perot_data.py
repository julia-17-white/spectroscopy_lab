import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lmfit as lm
import scipy
from scipy.optimize import curve_fit

def obtain_data(filename):
    '''
    Method using pandas read_csv function to obtain the oscilliscope data,
    clean it up, and plot it for preliminary analysis.

    Paraters:
        filename: the name of the csv file being read in.

    ReturnsL
        df: the dataframe containing the csv's data
    '''

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
    plt.ylabel('voltage (V)')
    # plt.plot(df['x-axis'], df['1'])

    return df

#################################################################
# using the scipy peak finding algorithm to locate the peaks
def peak_finding(df, peak_width, channel, neg: bool = False):
    '''
    Method employing scipy's signal's find_peaks algorithm to find the
    peaks of a dataframe. It then plots these peaks on to p of the data,
    allowign the user to determine its level of success.

    Parameters:
        df: the dataframe containing the data
        peak_width: allows you to specify the width of each peaks,
            controlling how many are obtained, etc.
        channel: specifies which column of the dataframe is being searched.
        neg: boolean specifying if we want to find maxima or minima

    Returns:
        peaks: the array containing all of the peaks and their associated info.
    '''

    if neg == True:
        x = -1
    else:
        x = 1
    peaks = scipy.signal.find_peaks(x*df[channel], width=peak_width)
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
# Defining a Number of Fitting Algorithms
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

    # print(fit_result.fit_report())
    
    return l_vals, slope, intercept

def pol_fitting(x_vals: list, y_vals: list):
    def func(x, a, b, c, d):
        return a + b * x + c * x ** 2 + d * x ** 3
    
    popt, _ = curve_fit(func, x_vals, y_vals)
    print(f'polynomial fit result: {popt}')
    # print(popt)
    x_fit = np.linspace(min(x_vals), max(x_vals), 500)
    y_fit = func(x_fit, *popt)
    return x_fit, y_fit

def gauss_fitting(x_vals: list, y_vals: list):
    # USE **KWARGS TO MAKE THIS MORE ROBUST IF DESIRED

    # define the Model
    g_model = lm.models.GaussianModel(independent_vars='x')

    # set initial parameters (optional step)
    g_model.set_param_hint('amplitude', value = 0.5)
    g_model.set_param_hint('center', value = 0)
    g_model.set_param_hint('sigma', value = 0.5)
    g_model.set_param_hint('gamma', value=-2)

    # create parameters
    my_params = g_model.make_params()

    # perform fit
    fit_result = g_model.fit(y_vals, x = x_vals, params = my_params)

    # acquiring data from the fit
    g_vals = fit_result.eval(x=x_vals)

    center = fit_result.params['center'].value
    center_unc = fit_result.params['center'].stderr
    height = fit_result.params['amplitude'].value
    fwhm = fit_result.params['fwhm'].value

    print('The center value is ' + str(center) + ' +/- ' + str(center_unc))

    # print(fit_result.fit_report())

    return g_vals, center, height, fwhm


#################################################################
# Building the claibration curve
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

    # time_diff = lin_fit[1] # 1/(the slope)
    # return time_diff

def fsr_calc(l, n):
    # need index of refraction of silicon at 1580nm
    # USE THIS CALCULATED FSR TO CALIBRATE THE Y-AXIS OF THE CALIBRATION PLOT
    fsr = (scipy.constants.c / (2 * l * n) ) * 1E-6 # reported in MHz
    print(f'the free spectral range is {fsr}')
    return fsr

def freq_calibration(fsr, delta_t, df):
    freq_cal = ((fsr)/delta_t)*df['x-axis'] #MHz

    # plt.figure()
    # plt.plot(freq_cal, df['1'])
    return freq_cal

#################################################################
# Calculating the Finesse

def fwhm_calc(df):
    # min_data = df
    # min_data['1'] = df['1']*-1
    mins = peak_finding(df, 1, '1', True)
    mid_point = int(len(mins[0])/2)
    data_min = mins[0][mid_point]
    data_max = mins[0][mid_point+1]
    one_peak_df = df.iloc[data_min:data_max]

    x_vals = one_peak_df['x-axis'].values
    y_vals = one_peak_df['1'].values

    x_shift = x_vals.mean()
    x_centered = x_vals - x_shift

    y_shift = y_vals.min()
    y_shifted = y_vals - y_shift

    plt.figure()
    plt.plot(x_centered, y_shifted)
    gauss_fit, center, height, fwhm = gauss_fitting(x_centered, y_shifted)
    plt.plot(x_centered, gauss_fit)

    return fwhm

def finesse_calc(fsr, fwhm, l):
    finesse_exp = fsr/fwhm
    finesse_the = ((100*l*fsr*1E6)/(2*(scipy.constants.c))) 
    return finesse_exp, finesse_the

#################################################################
# Calculating the Percentage of CO in the cell

def transmittance_calc(df, df2, delta_t, freq_cal):
    # print(freq_cal)
    transmittance = df['2']/df2['2']
    plt.figure()
    # plt.plot(x_vals, df['2'])
    # plt.plot(x_vals, df2['2'])
    plt.plot(freq_cal, transmittance)

    return transmittance


def phi_calc(freq_cal, gamma, P):
    phi_val = []
    for val in freq_cal:
        phi_val.append(float(1/np.pi) * ((gamma*P)/((gamma * P)**2 + val**2)))

    return phi_val


def percent_co(transmittance, freq_cal, n, l, S, phi):
    tau = -np.log(transmittance)
    print(tau)
    chi = []
    for i in range(len(phi)):
        chi.append(n*l*S*(phi[i]/tau.iloc[i]))
    plt.figure()
    plt.plot(freq_cal, chi)



def main():
    # defining necessary constants, etc.
    channel = '1'
    length = 16.483 * 10 # cm
    n = 3.45 # refractive index of Silicon https://srd.nist.gov/jpcrdreprint/1.555624.pdf
    gamma = 0.068 #broadening coefficient in 1/(cm * atm)
    P = 1 # atm
    S = 1.610E-23 # line strength in 1/cm

    # analyzing the data
    data_df = obtain_data('scope_5.csv')
    data2_df = obtain_data('scope_6.csv')
    peaks = peak_finding(data_df, 10, channel)
    time_diff1 = delta_t_calc(data_df, peaks)
    fsr = fsr_calc(length, n)
    fwhm = fwhm_calc(data_df)
    print(f'fwh: {fwhm:.4f}')
    freq_cal = freq_calibration(fsr, time_diff1, data_df)
    finesse_exp, finesse_theory = finesse_calc(fsr, fwhm, length)
    print(f'experimental finesse: {finesse_exp:.4f}')
    print(f'theoretical finesse: {finesse_theory:.4f}')
    # time_diff2 = delta_t_fit(data_df, peaks, fsr)
    print(f'time difference with no fitting: {time_diff1:.4f}s')
    # print(f'time difference with linear fitting: {time_diff2}')
    transmittance = transmittance_calc(data_df, data2_df, time_diff1, freq_cal)
    phi = phi_calc(freq_cal, gamma, P)
    percent_co(transmittance, freq_cal, n, length, S, phi)
    plt.show()


main()

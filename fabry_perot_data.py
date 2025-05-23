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
    # print(peaks)
    plt.figure()
    plt.plot(df['x-axis'], df[channel])
    plt.scatter(df['x-axis'][peaks[0] + 8], peaks[1]['prominences'] + 0.36, c = 'm', s=15)
    # the +8 is to account for the NaN indices which were removed
    # the +36 is a y-axis normalization
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')

    return(peaks)


#################################################################
# now i'll take the two middle peaks and calculate delta t with them
def delta_t_calc(df, peaks):
    '''
    Method finding the time difference between two consecutive peaks.

    Parameters:
        df: the dataframe containing the data.
        peaks: the array containing all of the peaks and their associated info.

    Returns:
        time_diff: the time difference between the successive peaks
    '''

    mid_point = int(len(peaks[0])/2)
    time_diff = df['x-axis'][peaks[0][mid_point]] - df['x-axis'][peaks[0][mid_point - 1]]
    print(f'the time difference is {time_diff:.4f} seconds')
    time_diff_unc = 0.0005 # s arbitrary value found by looking at the accuracy of the peaks array
        # measurements --> no proper uncertainty is provided
    return time_diff, time_diff_unc

#################################################################
# Defining a Number of Fitting Algorithms
def lin_fitting(x_vals: list, y_vals: list):
    '''
    Method to preform a linear fit on the data.
    '''

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
    '''
    Method to preform a polynomial x^3 fit on the data.
    '''

    def func(x, a, b, c, d):
        return a + b * x + c * x ** 2 + d * x ** 3
    
    popt, _ = curve_fit(func, x_vals, y_vals)
    # print(f'polynomial fit result: {popt}')
    # print(popt)
    x_fit = np.linspace(min(x_vals), max(x_vals), 500)
    y_fit = func(x_fit, *popt)
    return x_fit, y_fit

def lmfit_pol_fitting(x_vals: list, y_vals: list):
    '''
    Method to preform a polynomial x^3 fit on the data.
    '''

    def func(x, a, b, c, d):
        return a + b * x + c * x ** 2 + d * x ** 3
    
    params = lm.Parameters()
    
    #Note, the names here MUST match the parameters to your model function
    params.add('a', vary = True, value = 0, min = -1, max = 1)
    params.add('b', vary = True, value = 0, min = -1, max = 1)
    params.add('c', vary = True, value = 0, min = -1, max = 1)
    params.add('d', vary = True, value = 0, min = -1, max = 1)
    
    # define the Model
    g_model = lm.Model(func)

    # # set initial parameters (optional step)
    # g_model.set_param_hint('amplitude', value = 0.5)
    # g_model.set_param_hint('center', value = 0)
    # g_model.set_param_hint('sigma', value = 0.5)
    # g_model.set_param_hint('gamma', value=-2)

    # create parameters
    # my_params = g_model.make_params()

    # perform fit
    fit_result = g_model.fit(y_vals, x = x_vals, params = params)

    # acquiring data from the fit
    p_vals = fit_result.eval(x=x_vals)

    # print('The center value is ' + str(center) + ' +/- ' + str(center_unc))

    print(fit_result.fit_report())

    return p_vals


def gauss_fitting(x_vals: list, y_vals: list):
    '''
    Method to preform a Gaussian fit on the data.
    '''

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
    fwhm_unc = fit_result.params['fwhm'].stderr

    print('The center value is ' + str(center) + ' +/- ' + str(center_unc))

    # print(fit_result.fit_report())
    if fwhm_unc is None:
        fwhm_unc = 0.01*fwhm # very arbitrary but I have no other way to define it

    return g_vals, center, height, fwhm, fwhm_unc

def lorentz_fitting(x_vals: list, y_vals: list, N):
    '''
    Method to preform a Lorentzian fit on the data.
    '''

    def func(x, gamma, P, chi, N, L, S, y_norm, x0, a, b, c, d):
        # change to the n matching your simulation
        phi =  (1/np.pi) * ((gamma*P)/((gamma * P)**2 + (x-x0)**2))
        tau = chi * N * L * S * phi + y_norm

        bl = a + b*x + c*x**2 + d*x**3
        return tau + bl

    params = lm.Parameters()
    
    #Note, the names here MUST match the parameters to your model function
    params.add('gamma', vary = True, value = 0.068, min = 0, max = 3)
    params.add('P', vary = False, value = 1, min = 0, max = 100)
    params.add('chi', vary = True, value = 0.92, min = 0.9, max = 1)
    params.add('N', vary = True, value = N) # min = -100, max = 100
    params.add('L', vary = False, value = 40.3, min = -100, max = 100)
    params.add('S', vary = False, value = 1.610E-23, min = 0, max = 1)
    params.add('y_norm', vary = False, value=0)
    params.add('x0', vary = True, value = 58, min = 20, max = 80)
    params.add('a', vary = True, value = -0.00380328, min = -1, max = 1)
    params.add('b', vary = True, value = 9.7536e-04, min = -1, max = 1)
    params.add('c', vary = True, value = -1.8850e-05, min = -1, max = 1)
    params.add('d', vary = True, value = 9.0158e-08, min = -1, max = 1)
    
    # define the Model
    l_model = lm.Model(func)

    # # set initial parameters (optional step)
    # g_model.set_param_hint('amplitude', value = 0.5)
    # g_model.set_param_hint('center', value = 0)
    # g_model.set_param_hint('sigma', value = 0.5)
    # g_model.set_param_hint('gamma', value=-2)

    # create parameters
    # my_params = g_model.make_params()

    # perform fit
    fit_result = l_model.fit(y_vals, x = x_vals, params = params)

    # acquiring data from the fit
    t_vals = fit_result.eval(x=x_vals)
    chi = fit_result.params['chi'].value
    chi_unc = fit_result.params['chi'].stderr

    # print('The center value is ' + str(center) + ' +/- ' + str(center_unc))

    print(fit_result.fit_report())

    return t_vals, chi, chi_unc

# def lorentz_fitting(x_vals: list, y_vals: list):
#     '''
#     Method to preform a Lorentzian fit on the data.
#     '''

#     def func(x, gamma, P, chi, N, L, S, y_norm, x0):
#         # change to the n matching your simulation
#         phi =  (1/np.pi) * ((gamma*P)/((gamma * P)**2 + (x-x0)**2))
#         tau = chi * N * L * S * phi + y_norm
#         return tau

#     params = lm.Parameters()
    
#     #Note, the names here MUST match the parameters to your model function
#     params.add('gamma', vary = True, value = 0.068, min = 0, max = 3)
#     params.add('P', vary = False, value = 1, min = -100, max = 100)
#     params.add('chi', vary = True, value = 0.92, min = 0, max = 1)
#     params.add('N', vary = False, value = N, min = -100, max = 100)
#     params.add('L', vary = False, value = 40.3, min = -100, max = 100)
#     params.add('S', vary = True, value = 1.610E-23, min = 0, max = 1)
#     params.add('y_norm', vary = True, value=np.min(y_vals))
#     params.add('x0', vary = True, value = 60, min = 20, max = 80)
    
#     # define the Model
#     l_model = lm.Model(func)

#     # # set initial parameters (optional step)
#     # g_model.set_param_hint('amplitude', value = 0.5)
#     # g_model.set_param_hint('center', value = 0)
#     # g_model.set_param_hint('sigma', value = 0.5)
#     # g_model.set_param_hint('gamma', value=-2)

#     # create parameters
#     # my_params = g_model.make_params()

#     # perform fit
#     fit_result = l_model.fit(y_vals, x = x_vals, params = params)

#     # acquiring data from the fit
#     t_vals = fit_result.eval(x=x_vals)
#     chi = fit_result.params['chi'].value
#     chi_unc = fit_result.params['chi'].stderr

#     # print('The center value is ' + str(center) + ' +/- ' + str(center_unc))

#     print(fit_result.fit_report())

#     return t_vals, chi, chi_unc


#################################################################
# Building the claibration curve
def delta_t_fit(df, peaks, fsr):
    '''
    Method to develop the calibration curve for the delta_t
    to figure out how good of an approximation linearity is

    Plots:
        the delta_t vs fsr time stamps where fsr steps by one at each peak
        the linear fit of the center data points
        the polynomial fit of all the data points
    '''

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
    plt.ylabel('frequency (fsr)')
    plt.xlabel('time (s)')

    # time_diff = lin_fit[1] # 1/(the slope)
    # return time_diff

def fsr_calc(l, n, l_unc):
    '''
    Method to calculate the theoretical free spectral range using 
    the measured lengths and calculated n value
    '''

    # need index of refraction of silicon at 1580nm
    # USE THIS CALCULATED FSR TO CALIBRATE THE Y-AXIS OF THE CALIBRATION PLOT
    fsr = ((scipy.constants.c) / (2 * l * n) ) * 1E-9 # reported in GHz --> the 1E-1 converts the speed of light from m to cm
    fsr_unc = np.sqrt(((-(scipy.constants.c))/(2 * l**2 * n) * l_unc)**2) * 1E-9 # GHz
    print(f'the free spectral range is {fsr} +/- {fsr_unc}')
    return fsr, fsr_unc

def freq_calibration(fsr, delta_t, df):
    '''
    Method to build out the frequency calibration array.
    '''

    freq_cal = ((fsr)/delta_t)*df['x-axis'] #GHz

    # plt.figure()
    # plt.plot(freq_cal, df['1'])
    return freq_cal

#################################################################
# Calculating the Finesse

def fwhm_calc(df, fsr, delta_t):
    # min_data = df
    # min_data['1'] = df['1']*-1
    mins = peak_finding(df, 1, '1', True)
    mid_point = int(len(mins[0])/2)
    data_min = mins[0][mid_point]
    data_max = mins[0][mid_point+1]
    one_peak_df = df.iloc[data_min:data_max]

    x_vals = one_peak_df['x-axis'].values
    y_vals = one_peak_df['1'].values
    y_vals_c = ((fsr)/delta_t)*y_vals

    x_shift = x_vals.mean()
    x_centered = x_vals - x_shift

    y_shift = y_vals_c.min()
    y_shifted = y_vals_c - y_shift

    plt.figure()
    plt.plot(x_centered, y_shifted, label='data')
    gauss_fit, center, height, fwhm , fwhm_unc = gauss_fitting(x_centered, y_shifted)
    plt.plot(x_centered, gauss_fit, label='fit')
    plt.xlabel('frequency (GHz)')
    plt.ylabel('Voltage (V)')
    plt.legend()

    # print(f'fwhm unc = {fwhm_unc}')

    return fwhm, fwhm_unc

def finesse_calc(fsr, fwhm, l, fsr_unc, fwhm_unc, l_unc):
    finesse_exp = fsr/(fwhm * 10)
    # finesse_the = ((l*fsr*1E9)/(2*(scipy.constants.c))) 

    finesse_e_unc = np.sqrt(((1/fwhm)*fsr_unc)**2 + ((-fsr/(fwhm**2))*fwhm_unc)**2)*0.1
    # finesse_t_unc = (1E9/(2*scipy.constants.c))*np.sqrt((fsr*l_unc)**2 + (l*fsr_unc)**2)

    return finesse_exp, finesse_e_unc

#################################################################
# Calculating the Percentage of CO in the cell

def transmittance_calc(df, df2, freq_cal):
    '''
    Method to calculate the transmittance of light through the cell and
    thus find the absorption line. This uses both of the dataframes, dividing
    the light going through the cell by the light not going through the cell.
    '''
    # print(freq_cal)
    transmittance = df['2']/df2['2']
    # need to normalize the transmittances
    # subt_val = 1 - max(transmittance)
    transmittance = transmittance / np.max(transmittance)

    trans_unc = 0.001 # the data points don't come with uncertainties so we
    # will assume the uncertainty starting here

    plt.figure()
    # plt.plot(x_vals, df['2'])
    # plt.plot(x_vals, df2['2'])
    plt.plot(freq_cal, transmittance)
    plt.ylabel('transmittance')
    plt.xlabel('frequency difference (GHz)')

    return transmittance, trans_unc


# def phi_calc(freq_cal, gamma, P):
#     phi_val = []
#     for val in freq_cal:
#         phi_val.append(float(1/np.pi) * ((gamma*P)/((gamma * P)**2 + val**2)))

#     return phi_val


# def percent_co(transmittance, freq_cal, n, l, S, phi):
#     tau = -np.log(transmittance)
#     chi = []
#     for i in range(len(phi)):
#         chi.append(n*l*S*(phi[i]/tau.iloc[i]))
#     plt.figure()
#     plt.plot(freq_cal, chi)

#     chi_val = np.max(chi)
#     return chi_val

def tau_theoretical(nu, P, gamma, L, S, L_unc, P_unc):
    '''
    Method based off of the simulationscript to calculate the
    theoretical optical depth (tau value) of this setup as well
    as its uncertainty.
    '''

    r = 2.43 / 100 # m
    r_unc = 0.005 / 100 # m
    V = float(np.pi * r**2 * L) # m^3 
    V_unc = np.sqrt((2*np.pi*r*L*r_unc)**2 + (np.pi*r**2*L_unc)**2)
    print(f'the volume of the cell is {V} +/- {V_unc} m^3')
    k_erg = 1.380469E-16 # erg/K
    k = k_erg * (9.869E-7) # atm/K
    # k = 1.38e-23
    T = 293 # kelvin
    T_unc = 1
    # P = 101325
    n = P / (k * T)
    n_unc = np.sqrt((V/(k*T)*V_unc)**2 + ((-P*V)/(k*T**2)*T_unc)**2) # + (P/(k*T)*P_unc)**2
    # should i include a pressure uncertainty?
    # print(f'n = {n:.4f} +/- {n_unc:.4f}')
    chi = 1

    phi = [] # lineshape function
    tau = [] # optical depth
    tau_unc = []

    for i in range(len(nu)):
        phi_val = float((1/np.pi) * ((gamma*P)/((gamma * P)**2 + nu.iloc[i]**2)))
        phi.append(phi_val)
        tau_val = chi * n * L * S * phi_val #* 1E6
        tau_unc_val = np.sqrt( (chi*L*S*phi_val*n_unc)**2 + (chi*n*S*phi_val*L_unc)**2 )#* 1E6
        tau.append(tau_val)
        tau_unc.append(tau_unc_val)

    return tau, tau_unc

def tau_calc(transmittance, trans_unc):
    '''
    Method to calculate the experimental optical depth and its
    uncertainty using the experimentally measured transmittance values.
    '''

    tau = -np.log(transmittance)
    tau_unc = np.sqrt(((-1/transmittance)*trans_unc)**2)
    return tau, tau_unc

def chi_diff(tau_theory, tau_theory_unc, tau_exp, tau_exp_unc):
    '''
    Method comparing the experimental and theoretical optical
    depth values to obtain a value for the number density of
    carbon monoxide molecules in the cylinder.
    '''

    chi_diff = np.max(tau_exp) - np.max(tau_theory)
    chi_diff_unc = np.sqrt(np.max(tau_exp_unc)**2 + np.max(tau_theory_unc)**2)
    chi_exp = 1 - chi_diff

    return chi_exp, chi_diff_unc

def fit_trans(tau, freq_cal, L, N, L_unc, N_unc):
    # mask = (freq_cal < 50) | (freq_cal > 65)
    # imask = (freq_cal >= 50) & (freq_cal <= 65)
    # p_vals = lmfit_pol_fitting(freq_cal[mask], tau[mask])
    # # p_vals = p_vals.eval(x=freq_cal)
    # l_vals, chi, chi_unc = lorentz_fitting(freq_cal[imask], tau[imask])
    l_vals, chi, chi_unc = lorentz_fitting(freq_cal, tau, N)
    print(f'the fitted chi value is {chi:.4f} +/- {chi_unc:.4f}')
    chi_unc = np.sqrt(((1/N)*N_unc)**2 + ((1/L)*L_unc)**2)
    print(f'the fitted chi with propogated uncertainty is {chi:.4f} +/- {chi_unc:.4f}')

    # full_fit = np.empty_like(freq_cal)
    # full_fit[mask] = p_vals
    # full_fit[imask] = l_vals

    plt.figure()
    plt.plot(freq_cal, tau, label = 'data')
    # plt.plot(freq_cal[mask], p_vals, label = 'polynomial')
    # plt.plot(freq_cal[imask], l_vals, label = 'lorentzian')
    plt.plot(freq_cal, l_vals, c='m', label = 'fit')
    plt.legend()
    plt.xlabel('frequency (GHz)')
    plt.ylabel('optical depth')

def main():
    # defining necessary constants, etc.
    channel = '1'
    length = 16.483 / 1000 # m
    length_unc = 0.001 / 1000 # m
    cell_length = 40.3 / 100 # m
    cell_length_unc = 0.05 / 100 # m
    r = 2.43 / 10 # m
    V = float(np.pi * r**2 * cell_length) # m^3 
    k_erg = 1.380469E-16 # erg/K
    k = k_erg * (9.869E-7) # atm/K
    k = 1.38e-23
    T = 293 # K
    P = 0.97 # atm or 740  torr
    N = 101325 / (k * T)
    N_unc = 101325 * (1/(k * T**2)) * 1
    print(f'N = {N} +/- {N_unc}, bolzmann = {k}')
    n = 3.45 # refractive index of Silicon https://srd.nist.gov/jpcrdreprint/1.555624.pdf
    gamma_wvnmbr = 0.068 #broadening coefficient in 1/(cm * atm)
    gamma = (10)*scipy.constants.c*gamma_wvnmbr # Hz
    P_unc = 0.5 # atm
    S_wvnmbr = 1.610E-23 # line strength in 1/cm
    S = (10)*scipy.constants.c*S_wvnmbr

    # analyzing the data
    data_df = obtain_data('scope_5.csv')
    data2_df = obtain_data('scope_6.csv')
    peaks = peak_finding(data_df, 10, channel)
    time_diff1, time_diff_unc = delta_t_calc(data_df, peaks)
    fsr, fsr_unc = fsr_calc(length, n, length_unc)
    freq_cal = freq_calibration(fsr, time_diff1, data_df)
    fwhm, fwhm_unc = fwhm_calc(data_df, fsr, time_diff1)
    print(f'fwh: {fwhm:.4f}')
    finesse_exp, finesse_exp_unc = finesse_calc(fsr, fwhm, length, fsr_unc, fwhm_unc, length_unc)
    print(f'experimental finesse: {finesse_exp:.4f} +/- {finesse_exp_unc}')
    # time_diff2 = delta_t_fit(data_df, peaks, fsr)
    print(f'time difference with no fitting: {time_diff1:.4f}s')
    # print(f'time difference with linear fitting: {time_diff2}')
    transmittance, transmittance_unc = transmittance_calc(data_df, data2_df, freq_cal)
    # phi = phi_calc(freq_cal, gamma, P)
    # chi = percent_co(transmittance, freq_cal, n, cell_length, S, phi)
    # print(f'chi = {chi}')

    tau_theory, tau_theory_unc = tau_theoretical(freq_cal, P, gamma, cell_length, S, cell_length_unc, P_unc)
    tau_exp, tau_exp_unc = tau_calc(transmittance, transmittance_unc)
    print(f'theoretical tau value: {np.max(tau_theory)} +/- {np.max(tau_theory_unc)}') # i think something is wrong with this tau value
    print(f'experimental tau value: {np.max(tau_exp):.4f} +/- {np.max(tau_exp_unc):.4f}')

    chi_res, chi_unc = chi_diff(tau_theory, tau_theory_unc, tau_exp, tau_exp_unc)
    print(f'measured chi value: {chi_res} +/- {chi_unc}')

    delta_t_fit(data_df, peaks, fsr)

    fit_trans(tau_exp, freq_cal, cell_length, N, cell_length_unc, N_unc)

    plt.show()


main()

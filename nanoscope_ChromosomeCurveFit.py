import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from Nanoscope_converter import nanoscope_converter
import numpy as np
import pandas as pd
#import seaborn as sns
from scipy.optimize import curve_fit, leastsq
import os.path
from numpy.linalg import norm
from scipy.signal import savgol_filter  # Savitzky Golay filter for smoothing data with too much noise
import glob  # this is for listing all the files in a given directory
from nanoscope_CurvesContactPoint_determination import contact_pointFinder1, contact_pointFinder2
from scipy.integrate import simps



def fit_func_linear(x, a, b):  # sensitivity is corrected by applying a linear fit to the last part of the curve
    return a * x + b

def indentation_fit(path):
    # ---- HOW MANY POINTS TO USE FOR CORRECTING THE SENSITIVITY? ----
    # points = input('How many points to use for correcting the sensitivity? Type "no" for avoid sensitivity correction ')
    # points = 100
    points = 'no'
    curves_name = []
    Young_modulus = []
    Young_modulus_TipC = []
    all_curves = []
    cpoints = []
    ALL_forw_vel = []
    ALL_ind_rate = []
    RSE_values = []
    RSE_values_TipC = []
    ALL_x_pos = []
    ALL_y_pos = []
    ALL_fit_sep = []
    ALL_fit = []
    ALL_fit_sep_TipC = []
    ALL_fit_TipC = []
    curve_data = nanoscope_converter(path)
    raw_separation = curve_data[0]  # script per lettura e conversione dei file Bruker
    force = curve_data[1]  # script per lettura e conversione dei file Bruker
    forw_vel = curve_data[10]  # forward indenting velocity in nm/s
    ALL_forw_vel.append(forw_vel)
    ind_rate = curve_data[11]  # indentation rate in Hz
    ALL_ind_rate.append(ind_rate)
    scan_size = curve_data[12]  # scan size
    x_pos = curve_data[13]
    y_pos = curve_data[14]
    ALL_x_pos.append(x_pos)
    ALL_y_pos.append(y_pos)
    # -------CORRECTION FOR SENSITIVITY-------
    if points == 'no':
        separation = raw_separation
    else:
        points = int(points)
        corrected_separation = raw_separation[(len(raw_separation) - points):]
        corrected_force = force[(len(force) - points):]
        fit_param = curve_fit(fit_func_linear, corrected_separation, corrected_force, maxfev=100000)
        correction = (force - fit_param[0][1]) / fit_param[0][0]
        separation = raw_separation - correction
    # ---- CORRECTION FOR BASELINE NOT HORIZONTAL ----
    n = 1000
    new_sep = raw_separation[:n]
    new_force = force[:n]
    fit_param = curve_fit(fit_func_linear, new_sep, new_force, maxfev=100000)
    force = force - (fit_param[0][0] * raw_separation + fit_param[0][1])
    # ------SAVITSKY GOLAY FILTERING IN THE CASE THE SIGNAL PRESENTS TOO MUCH NOISE
    force = savgol_filter(force, 35, 3, mode='nearest')
    # ---- CONTACT POINT DETERMINATION ----
    contact = contact_pointFinder2(separation, force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    cpoints.append(cp_x)
    force = force - cp_y
    # Calculate the area below the indentation curve
    area_simps = simps(force, dx=np.abs(separation[1] - separation[0]))
    # Shifting the y coordinate of the contact point to 0 (and shifting up the whole curve) in order
    # not to compromise the fit when the force is below zero due to instabilities of the baseline
    # ---- FITTING OF THE DELIMITED PART OF THE CURVE ----
    # define the part of the curve to be fitted (subtracting cp_x because the x coordinate
    # should be zero at the contact pt., i.e. when the fit starts)
    # fitting a window that is n% of the total indentation (cp_x - n/100 * cp_x)
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - 0.3 * cp_x]
    fit_separation = separation[cp_index: end_pt[0]] - cp_x
    fit_force = force[cp_index: end_pt[0]]
    R_tip = 10  # tip radius--> in future better to insert as input request
    nu = 0.5  # Poisson coefficient -->  either 0.5 or 0.7
    t = cp_x  # thickness in nm --> set equal to the contact point x coordinate

    # ---- Standard Hertz fit model ----
    def fit_func_hertz(x, E):
        f = (4 / 3) * (R_tip ** 0.5) * (E / (1 - (nu) ** 2)) * (np.abs(x) ** 1.5)
        return f

    # ---- Modified Hertz fit (from Dimitriadis et al. Biophysical journal 82.5 (2002): 2798-2810.) ----
    # #      FOR SAMPLES NOT BONDED TO THE SURFACE WHICH CAN SLIP (E is in MPa)
    def fit_func_modifhertz_nb(x, E):
        f = (16 / 9) * E * ((R_tip) ** 0.5) * (np.abs(x) ** 1.5) * ((
                1 + 0.884 * ((R_tip * np.abs(x)) ** 0.5) / t + 0.781 * (
                ((R_tip * np.abs(x)) ** 0.5) / t) ** 2 + 0.386 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 3 + 0.0048 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 4))
        return f

    # ---- (Modified Hertz fit from Dimitriadis et al. Biophysical journal 82.5 (2002): 2798-2810.) ----
    # #      FOR SAMPLES BONDED TO THE SURFACE WHICH CANNOT SLIP (E is in MPa)
    def fit_func_modifhertz_b(x, E):
        f = (16 / 9) * E * ((R_tip) ** 0.5) * (np.abs(x) ** 1.5) * ((
                1 + 1.133 * ((R_tip * np.abs(x)) ** 0.5) / t + 1.283 * (
                ((R_tip * np.abs(x)) ** 0.5) / t) ** 2 + 0.769 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 3 + 0.0975 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 4))
        return f

    # ---- Corrected fit for conical tips (from Garcia R. et al. Biophysical journal 114.12 (2018): 2923-2932) ----
    #  h is the sample thickness (contact point) and theta is the cone angle
    def fit_func_garcia_conicalTip(x, E):
        theta = np.deg2rad(35)
        f0 = (8 * np.tan(theta) * E * x ** 2) / (3 * np.pi)
        f = f0 * (1 + (0.721 * x * np.tan(theta) / t) + (0.650 * x ** 2 * (np.tan(theta)) ** 2 / t ** 2) + (
                    0.491 * x ** 3 * (np.tan(theta)) ** 3 / t ** 3) + (
                              0.225 * x ** 4 * (np.tan(theta)) ** 4 / t ** 4))
        return f

    # simple exponential function
    def fit_HertzEffectiveCP(x, E):
        params, pcov = curve_fit(selected_fit, fit_separation, fit_force, p0=[0.01], bounds=(0, 10000),
                                  maxfev=100000000)
        fit = selected_fit(fit_separation, params[0])
        # calculate residuals
        res = fit_force - fit
        sum_res = np.sum(res ** 2)
        print(sum_res)
        print("E= ", params[0]*1000)



    # ---- FITTING THE INDENTATION CURVE ----
    # (if wants to fit with moving window, see end of the script)
    fitting_models = [fit_func_hertz, fit_func_modifhertz_nb, fit_func_modifhertz_b,
                      fit_func_garcia_conicalTip, fit_HertzEffectiveCP]
    # ---- SELECT THE FITTING MODEL -----
    selected_fit = fitting_models[2]
    E_bis = (15 / 8) * (1 - nu ** 2) * (cp_x ** (-5 / 2)) * area_simps / (np.sqrt(R_tip))
    params, pcov = curve_fit(selected_fit, fit_separation, fit_force, p0=[0.001], bounds=(0, 10000),
                             maxfev=100000000)
    fit = selected_fit(fit_separation, params[0])
    young_modulus = params[0] * 1000
    try:
        # params, pcov = curve_fit(selected_fit, fit_separation, fit_force)
        # fit = selected_fit(fit_separation, params[0], params[1])
        # Young modulus in kPa
        # print('Young Modulus (kPa): ', young_modulus)
        # R square, taken from Lin, David C., et al. "Spherical indentation of soft matter beyond the Hertzian regime:
        # numerical and experimental validation of hyperelastic models." Biomechanics and modeling in mechanobiology 8.5
        # (2009): 345-358
        error = 1 - (norm(fit_force - fit) ** 2 / norm(fit_force - np.mean(fit_force)) ** 2)
        # print('R square: ', R2)
    except:
        # fit = [0]*len(fit_separation)
        # young_modulus = 0
        error = 0
        # print('Fit error')

    # params, pcov = curve_fit(selected_fit, fit_separation, fit_force, p0=[0.01], bounds=(0, 10000), maxfev=100000000)
    # fit = selected_fit(fit_separation, params[0])
    # Young modulus in kPa
    # young_modulus = params[0]*1000
    #print('Young Modulus (kPa): ', young_modulus)
    # print('Young modulus: %f Covariance: %f ' % (young_modulus, std_dev))

    # ---- PLOTTING ----
    # plt.ion()
    # if error < 0:
    #     print(error)
    #     fig, ax = plt.subplots()
    #     ax.grid()
    #     ax.scatter(separation, force, s=1, c='gray', alpha=0.5)
    #     ax.scatter(cp_x, cp_y, s=25, c='k', marker='X')
    #     ax.set_xlabel('Separation (nm)')
    #     ax.set_ylabel('Force (pN)')
    #     # ---- add cp_x to shift the fit on the old curve where zero is the surface ----
    #     ax.plot(fit_separation + cp_x, fit, c='r')
    #     plt.show()
    # ax.plot(fit_separation + cp_x, fit + RSE, c='g')
    # ax.plot(fit_separation + cp_x, fit - RSE, c='g')
    # plt.show()
    # plt.pause(0.7)
    # plt.close()
    # print(type(x_pos), type(error))
    return x_pos, y_pos, young_modulus, error, cp_x, area_simps, E_bis

def area_viscoelasticity(path):
    # ---- HOW MANY POINTS TO USE FOR CORRECTING THE SENSITIVITY? ----
    # points = input('How many points to use for correcting the sensitivity? Type "no" for avoid sensitivity correction ')
    # points = 100
    points = 'no'
    curves_name = []
    all_curves = []
    cpoints = []
    rt_cpoints = []
    ALL_x_pos = []
    ALL_y_pos = []

    curve_data = nanoscope_converter(path)
    raw_separation = curve_data[0]  # script per lettura e conversione dei file Bruker
    force = curve_data[1]  # script per lettura e conversione dei file Bruker
    rt_raw_separation = curve_data[2]  # script per lettura e conversione dei file Bruker
    rt_force = curve_data[3]  # script per lettura e conversione dei file Bruker
    x_pos = curve_data[13]
    y_pos = curve_data[14]
    ALL_x_pos.append(x_pos)
    ALL_y_pos.append(y_pos)
    # -------CORRECTION FOR SENSITIVITY-------
    if points == 'no':
        separation = raw_separation
        rt_separation = rt_raw_separation
    else:
        points = int(points)
        corrected_separation = raw_separation[(len(raw_separation) - points):]
        corrected_force = force[(len(force) - points):]
        fit_param = curve_fit(fit_func_linear, corrected_separation, corrected_force, maxfev=100000)
        correction = (force - fit_param[0][1]) / fit_param[0][0]
        separation = raw_separation - correction
    # ---- CORRECTION FOR BASELINE NOT HORIZONTAL ----
    n = 1000
    new_sep = raw_separation[:n]
    new_force = force[:n]
    fit_param = curve_fit(fit_func_linear, new_sep, new_force, maxfev=100000)
    force = force - (fit_param[0][0] * raw_separation + fit_param[0][1])
    # ------SAVITSKY GOLAY FILTERING IN THE CASE THE SIGNAL PRESENTS TOO MUCH NOISE
    force = savgol_filter(force, 35, 3, mode='nearest')
    # ---- CONTACT POINT DETERMINATION ----
    contact = contact_pointFinder2(separation, force)
    rt_contact = contact_pointFinder2(rt_separation, rt_force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    rt_cp_x, rt_cp_y, rt_cp_index = rt_contact[0], rt_contact[1], rt_contact[2]
    cpoints.append(cp_x)
    rt_cpoints.append(rt_cp_x)
    # Calculate the area below the indentation curve
    area_simps = simps(force[cp_index:], dx=np.abs(separation[1] - separation[0]))
    # Calculate the area below the retraction curve
    rt_area_simps = simps(rt_force[rt_cp_index:], dx=np.abs(separation[1] - separation[0]))
    # ---- Calculating the index of plasticity ---- from Klymenko, O., et al. "Energy dissipation in the AFM
    # elasticity measurements." Acta Physica Polonica-Series A General Physics 115.2 (2009): 548.
    a1 = area_simps
    a2 = rt_area_simps
    eta = 1 - (a2/a1)
    #
    # fig, ax = plt.subplots()
    # ax.grid()
    # ax.scatter(separation, force, s=1, c='gray', alpha=0.5)
    # ax.scatter(rt_separation, rt_force, s=1, c='gray', alpha=0.5)
    # ax.scatter(cp_x, cp_y, s=25, c='k', marker='X')
    # ax.scatter(rt_cp_x, rt_cp_y, s=25, c='orange', marker='X')
    # ax.set_xlabel('Separation (nm)')
    # ax.set_ylabel('Force (pN)')
    # plt.show()
    return x_pos, y_pos, eta, cp_x


# path = r"E:\20221115_Chromosomes_PABuffer_IndentationTests_SNL10C\PS150452\cr00004.000"
# indentation_fit(path)

# ---- procedure for fitting with a moving window ----
    '''for i in range(len(separation[cp_index: -300])):
        print(separation[end_pt[i]])
        fit_separation = separation[cp_index+i: end_pt[i]] - cp_x
        fit_force = force[cp_index+i: end_pt[i]]
        fitting_models = [fit_func_hertz, fit_func_modifhertz_nb, fit_func_modifhertz_b,
                          fit_func_garcia_conicalTip, fit_HertzEffectiveCP]
        # ---- SELECT THE FITTING MODEL -----
        selected_fit = fitting_models[2]
        params, pcov = curve_fit(selected_fit, fit_separation, fit_force, p0=[0.01], bounds=(0, 10000),
                                  maxfev=100000000)
        fit = selected_fit(fit_separation, params[0])
        # Young modulus in kPa
        young_modulus = params[0] * 1000
        print('Young Modulus (kPa): ', young_modulus)
        # calculate residuals
        res = fit_force - fit
        sum_res = np.sum(res ** 2)
        print("Sum of residuals = ", sum_res)
        print("E= ", params[0] * 1000)

        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(separation, force, s=1, c='gray', alpha=0.5)
        ax.scatter(cp_x, cp_y, s=25, c='k', marker='X')
        ax.set_xlabel('Separation (nm)')
        ax.set_ylabel('Force (pN)')
        # ---- add cp_x to shift the fit on the old curve where zero is the surface ----
        ax.plot(fit_separation + cp_x, fit, c='r')
        # ax.plot(fit_separation + cp_x, fit + RSE, c='g')
        # ax.plot(fit_separation + cp_x, fit - RSE, c='g')
        plt.show() '''

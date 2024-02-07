import matplotlib.pyplot as plt
from Nanoscope_converter import nanoscope_converter
import numpy as np
from scipy.optimize import curve_fit
from numpy.linalg import norm
from scipy.signal import savgol_filter  # Savitzky Golay filter for smoothing data with too much noise
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
from scipy.integrate import simps


'''This script contains two main custom functions for different analyses of the AFM force curves:
    1) indentation_fit for fitting the indenting curve
    2) area_viscoelasticity for the analysis of the area under the indenting and retracting curve.'''


# function for linear fit
def fit_func_linear(x, a, b):
    return a * x + b


def indentation_fit(path):
    '''Fit the approach curve with different contact mechanics models'''
    # Reading the curve data
    curve_data = nanoscope_converter(path)
    raw_separation = curve_data[0]
    force = curve_data[1]
    # forw_vel = curve_data[10]
    # ind_rate = curve_data[11]
    # scan_size = curve_data[12]
    x_pos = curve_data[13]
    y_pos = curve_data[14]
    # In the case the sensitivity was not correctly calibrated in the instrument
    # fit the part of curve describing surface contact and puts it vertical straight
    # points = 100 # n. of points to use for the fit
    points = 'no'
    if points == 'no':
        separation = raw_separation
    else:
        points = int(points)
        corrected_separation = raw_separation[(len(raw_separation) - points):]
        corrected_force = force[(len(force) - points):]
        fit_param, pcov = curve_fit(fit_func_linear, corrected_separation, corrected_force, maxfev=100000)
        correction = (force - fit_param[1]) / fit_param[0]
        separation = raw_separation - correction
    # Sometimes (due to instrumental errors) both approaching and
    # retracting curves are not perfectly horizontal, correction is similar to sensitivity
    # points to use to level the curves (first # points in the baseline part)
    n = 1000
    fit_param, pcov = curve_fit(fit_func_linear, raw_separation[:n], force[:n], maxfev=100000)
    force = force - (fit_param[0] * raw_separation + fit_param[1])
    # Further filtering when the signal is particularly noisy
    force = savgol_filter(force, 35, 3, mode='nearest')
    # Contact point determination
    contact = contact_pointFinder2(separation, force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    # setting the value of force at the contact point exactly to zero helps the fit quality
    force = force - cp_y
    # Fitting: data preparation
    # should be zero at the contact pt., i.e. when the fit starts) setting the contact point ascissa as zero and
    # fitting a portion of curve n times the total indentation (cp_x - n * cp_x)
    n = .3
    # index of last point of the fitted section
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - n * cp_x]
    # portion of curve to be fitted
    fit_separation = separation[cp_index: end_pt[0]] - cp_x
    fit_force = force[cp_index: end_pt[0]]
    # Properties of the tip (only used in Hertz models)
    # tip radius (nm)
    R_tip = 10
    # Poisson coefficient
    nu = 0.5
    # thickness of the sample (nm), set equal to the contact point x coordinate
    t = cp_x

    # Fitting functions:
    def fit_func_hertz(x, E):
        '''Classical Hertz fit model.'''
        f = (4 / 3) * (R_tip ** 0.5) * (E / (1 - (nu) ** 2)) * (np.abs(x) ** 1.5)
        return f

    def fit_func_modifhertz_nb(x, E):
        '''Modified Hertz fit from Dimitriadis et al. Biophysical journal 82.5 (2002): 2798-2810.
        FOR SAMPLES NOT BONDED TO THE SURFACE WHICH CAN SLIP (E is in MPa).'''
        f = (16 / 9) * E * ((R_tip) ** 0.5) * (np.abs(x) ** 1.5) * ((
                1 + 0.884 * ((R_tip * np.abs(x)) ** 0.5) / t + 0.781 * (
                ((R_tip * np.abs(x)) ** 0.5) / t) ** 2 + 0.386 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 3 + 0.0048 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 4))
        return f

    def fit_func_modifhertz_b(x, E):
        '''Modified Hertz fit from Dimitriadis et al. Biophysical journal 82.5 (2002): 2798-2810.
        FOR SAMPLES BONDED TO THE SURFACE WHICH CANNOT SLIP (E is in MPa).'''
        f = (16 / 9) * E * ((R_tip) ** 0.5) * (np.abs(x) ** 1.5) * ((
                1 + 1.133 * ((R_tip * np.abs(x)) ** 0.5) / t + 1.283 * (
                ((R_tip * np.abs(x)) ** 0.5) / t) ** 2 + 0.769 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 3 + 0.0975 * (
                        ((R_tip * np.abs(x)) ** 0.5) / t) ** 4))
        return f

    def fit_func_garcia_conicalTip(x, E):
        '''Corrected fit for conical tips (from Garcia R. et al. Biophysical journal 114.12 (2018): 2923-2932). h is
        the sample thickness (contact point) and theta is the cone angle.'''
        theta = np.deg2rad(35)
        f0 = (8 * np.tan(theta) * E * x ** 2) / (3 * np.pi)
        f = f0 * (1 + (0.721 * x * np.tan(theta) / t) + (0.650 * x ** 2 * (np.tan(theta)) ** 2 / t ** 2) + (
                0.491 * x ** 3 * (np.tan(theta)) ** 3 / t ** 3) + (
                          0.225 * x ** 4 * (np.tan(theta)) ** 4 / t ** 4))
        return f


    # Applying the fit to the indentation portion
    # (if wants to fit with moving window, see end of the script)
    fitting_models = [fit_func_hertz, fit_func_modifhertz_nb, fit_func_modifhertz_b,
                      fit_func_garcia_conicalTip]
    # Selecting which model to use for the fit
    selected_fit = fitting_models[2]
    params, pcov = curve_fit(selected_fit, fit_separation, fit_force,
                             p0=[0.001], bounds=(0, 10000), maxfev=100000000)
    fit = selected_fit(fit_separation, *params)
    # Young Modulus in kPa
    young_modulus = params[0] * 1000
    try:
        # R square, taken from Lin, David C., et al. "Spherical indentation of soft matter beyond the Hertzian regime:
        # numerical and experimental validation of hyperelastic models." Biomechanics and modeling in mechanobiology 8.5
        # (2009): 345-358
        error = 1 - (norm(fit_force - fit) ** 2 / norm(fit_force - np.mean(fit_force)) ** 2)
        # print('R square: ', R2)
    except:
        # when the fit does not converge (or is bad) r-squared cannot be calculated correctly, in these cases its
        # value is set to 0 to be discarded later
        error = 0
    # Calculate Young Modulus from the area under the indenting curve
    area_simps = simps(force, dx=np.abs(separation[1] - separation[0]))
    # the area under the indenting curve (indentation energy) is equal to the integral of the force(described by
    # Hertz equation). From that we extract the Young Modulus as E_bis
    E_bis = (15 / 8) * (1 - nu ** 2) * (cp_x ** (-5 / 2)) * area_simps / (np.sqrt(R_tip))

    # Plotting the curve
    if error == 0:
        print(error)
    else:
        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(separation, force, s=5, c='gray', alpha=0.5, zorder=1, label='Force curve')
        ax.plot(fit_separation + cp_x, fit, c='r', linestyle='--', zorder=2, label='Fit')
        ax.scatter(cp_x, cp_y - cp_y, s=25, c='k', marker='X', zorder=3, label='Contact point')
        ax.set_xlabel('Separation (nm)')
        ax.set_ylabel('Force (pN)')
        plt.legend()
        plt.savefig("nanoscope_CurveFit_FC_Chromosome_Modified_Hertz_Fit_example.png", dpi=300)
        plt.show()

    return x_pos, y_pos, young_modulus, error, cp_x, area_simps, E_bis


def area_viscoelasticity(path):
    '''Calculates the "viscoelasticity index", a parameter that indicates the energy dissipation within the loading
    and unloading regimes (i.e., where forces > 0). The function reads the curve file, applies baseline and sensitivity
    corrections and estimate the index by calculating the areas underneath the two curves.'''
    # Reading curve data
    curve_data = nanoscope_converter(path)
    # NON-filtered separation and force for approach and retraction curves
    raw_separation = curve_data[0]
    force = curve_data[1]
    rt_raw_separation = curve_data[2]
    rt_force = curve_data[3]
    # xy coordinates for the indentation point
    x_pos = curve_data[13]
    y_pos = curve_data[14]
    # In the case the sensitivity was not correctly calibrated in the instrument
    # fit the part of curve describing surface contact and puts it vertical straight
    # points = 100 # n. of points to use for the fit
    points = 'no'
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
    # Sometimes (due to instrumental errors) both approaching and
    # retracting curves are not perfectly horizontal, correction is similar to sensitivity
    # points to use to level the curves (first # points in the baseline part)
    n = 1000
    new_sep = raw_separation[:n]
    new_force = force[:n]
    fit_param = curve_fit(fit_func_linear, new_sep, new_force, maxfev=100000)
    force = force - (fit_param[0][0] * raw_separation + fit_param[0][1])
    rt_force = rt_force - (fit_param[0][0] * rt_separation + fit_param[0][1])
    # Further filtering when the signal is particularly noisy
    force = savgol_filter(force, 35, 3, mode='nearest')
    rt_force = savgol_filter(rt_force, 35, 3, mode='nearest')
    # Contact point determination
    contact = contact_pointFinder2(separation, force)
    # also finding the point from which the retraction force < 0
    rt_contact = contact_pointFinder2(rt_separation, rt_force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    rt_cp_x, rt_cp_y, rt_cp_index = rt_contact[0], rt_contact[1], rt_contact[2]
    # Calculate the area below the indentation curve (where force > 0)
    area_simps = simps(force[cp_index:], dx=np.abs(separation[1] - separation[0]))
    # Calculate the area below the retraction curve (where force > 0)
    rt_area_simps = simps(rt_force[rt_cp_index:], dx=np.abs(separation[1] - separation[0]))
    # ---- Calculating the "index of plasticity" from Klymenko, O., et al. "Energy dissipation in the AFM elasticity
    # measurements." Acta Physica Polonica-Series A General Physics 115.2 (2009): 548.
    eta = 1 - (rt_area_simps / area_simps)
    print("Viscoelasticity index: ", eta)
    # Plotting
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(separation, force, s=1, c='b', alpha=0.5, zorder=2, label='Indenting Force Curve')
    ax.scatter(rt_separation, rt_force, s=1, c='r', alpha=0.5, zorder=2, label='Retracting Force Curve')
    ax.scatter(cp_x, cp_y, s=25, c='green', marker='X', zorder=3, label='Contact point')
    ax.scatter(rt_cp_x, rt_cp_y, s=25, c='k', marker='X', zorder=3, label='Starting of pulling regime')
    # Coloring area under the curves
    ax.fill_between(separation[cp_index:], force[cp_index:], 0,
                         color='dodgerblue', alpha=0.3, hatch='\\', zorder=1, label='Loading energy')
    ax.fill_between(rt_separation[rt_cp_index:], rt_force[rt_cp_index:], 0,
                    color='salmon', alpha=0.3, hatch='//', zorder=1, label='Unloading energy')
    ax.set_xlabel('Separation (nm)')
    ax.set_ylabel('Force (pN)')
    plt.legend()
    plt.savefig("nanoscope_CurveFit_FC_Chromosome_ViscIndex_example.png", dpi=300)
    plt.show()

    return x_pos, y_pos, eta, cp_x


# Testing the two functions with a random force curve
area_viscoelasticity("Example_RawData/cr00037.000")
indentation_fit("Example_RawData/cr00037.000")
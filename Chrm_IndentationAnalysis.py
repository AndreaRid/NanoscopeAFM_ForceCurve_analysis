import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter  # Savitzky Golay filter for smoothing data with too much noise
from scipy.interpolate import UnivariateSpline


# function for linear fit
def fit_func_linear(x, a, b):
    return a * x + b


folder = "PS150452"
# storing the directories of all the files
all_files = [f for f in glob.glob(folder + '**/*', recursive=True) if '.000' in f]
# image of the chromosome is the only .tif file in the folder
img_path = [item for item in all_files if '.tif' in item]

# fig, ax = plt.subplots(1, 2)
for curve in all_files:
    print(curve)
    curve_data = nanoscope_converter(curve)
    separation = curve_data[6]
    force = curve_data[7]
    # Sometimes (due to instrumental errors) both approaching and
    # retracting curves are not perfectly orizontal horizontal, correction is similar to sensitivity
    # points to use to level the curves (first # points in the baseline part)
    n = 1000
    fit_param, pcov = curve_fit(fit_func_linear, separation[:n], force[:n], maxfev=100000)
    force = force - (fit_param[0] * separation + fit_param[1])
    # Contact point determination
    contact = contact_pointFinder2(separation, force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    if cp_x <= 150:
        continue
    force = force - cp_y
    # Fitting: data preparation
    # should be zero at the contact pt., i.e. when the fit starts) setting the contact point ascissa as zero and
    # fitting a portion of curve n times the total indentation (cp_x - n * cp_x)
    n = .5
    # index of last point of the fitted section
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - n * cp_x]
    # portion of curve to be fitted
    fit_separation = separation[cp_index: end_pt[0]] - cp_x
    fit_force = force[cp_index: end_pt[0]]
    fit_separation = fit_separation * -1
    fit_separation = sorted(fit_separation)
    # Using spline interpolation to obtain a smoother trace
    spline_interp = UnivariateSpline(fit_separation, fit_force, k=1, s=55)
    # Generating finer x values for smooth plot
    fit_separation_spline = np.linspace(min(fit_separation), max(fit_separation), 500)
    # Interpolating y values using the spline interpolation function
    fit_force_spline = spline_interp(fit_separation_spline)
    df_dx = np.gradient(fit_force_spline, fit_separation_spline)
    df_dx = savgol_filter(df_dx, 205, 3, mode='nearest')
    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(fit_separation, fit_force, alpha=0.5, c='gray')
    # ax[0].scatter(fit_separation_spline, fit_force_spline, s=4)
    # ax[0].set_yscale("log")
    # ax[0].set_xscale("log")
    # ax[1].scatter(fit_separation_spline, df_dx, s=4)


plt.show()

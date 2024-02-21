import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter  # Savitzky Golay filter for smoothing data with too much noise
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps   # Calculating the area under the curve


'''This script processes all the indentation curves within the "Chromosome" folder and calculates useful descriptors 
such as the min and max values for the first derivative, the max indenting force, the contct point and the 
indentation energy in order to develop clustering analyses in later steps.'''


# function for linear fit
def fit_func_linear(x, a, b):
    return a * x + b



folders = [f for f in glob.glob("Example_RawData/Chromosomes" + '**/*', recursive=True)]
# folder = "Chromosome_PS161143"
# storing the directories of all the files
all_files = []
for folder in folders:
    all_file = [f for f in glob.glob(folder + '**/*', recursive=True) if '.000' in f]
    all_files.extend(all_file)
# image of the chromosome is the only .tif file in the folder
img_path = [item for item in all_files if '.tif' in item]

cp_x_values = []
ind_areas = []
df = pd.DataFrame()
all_curves = []
curves_df = pd.DataFrame()

# For plotting all the curves together, activate the first plt.subplots()
# fig, ax = plt.subplots(1, 2)
curves_array = [[] for _ in range(len(all_files))]
curves_array_xaxis = [[] for _ in range(len(all_files))]
for i, curve in enumerate(all_files):
    # name of the curve
    curve_name = curve[-11:]
    print("processing curve: ", curve_name)
    # Plotting each single curve independently
    # fig, ax = plt.subplots(1, 2)
    # plt.ion()
    # Processing and extracting curve data
    curve_data = nanoscope_converter(curve)
    separation = curve_data[6]
    force = curve_data[7]
    # Sometimes (due to instrumental errors) both approaching and
    # retracting curves are not perfectly horizontal, correction is similar to sensitivity
    # points to use to level the curves (first # points in the baseline part)
    n = 1000
    fit_param, pcov = curve_fit(fit_func_linear, separation[:n], force[:n], maxfev=100000)
    force = force - (fit_param[0] * separation + fit_param[1])
    # Contact point determination
    contact = contact_pointFinder2(separation, force)
    cp_x, cp_y, cp_index = contact[0], contact[1], contact[2]
    if cp_x <= 150:
        continue
    cp_x_values.append(cp_x)
    force = force - cp_y
    all_curves.append(curve)

    # Data preparation: setting the contact point  (i.e. where the fit starts) abscissa as zero and selecting
    # only a portion of curve of size n-times the total indentation (cp_x - n * cp_x)
    n = .85
    # index of last point of the fitted portion
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - n * cp_x]
    # portion of curve to be analyzed
    roi_separation = separation[cp_index: end_pt[0]] - cp_x
    roi_force = force[cp_index: end_pt[0]]
    roi_separation = roi_separation * -1
    roi_separation = sorted(roi_separation)
    # Using spline interpolation to obtain a smoother trace
    spline_interp = UnivariateSpline(roi_separation/np.max(roi_separation), roi_force, k=3, s=15)
    # Generating finer x values for smooth plot
    # roi_separation_spline = np.linspace(min(roi_separation), max(roi_separation), 500)
    roi_separation_spline = np.linspace(0, 1, 505)
    # Interpolating y values using the spline interpolation function
    roi_force_spline = spline_interp(roi_separation_spline)
    # plt.plot(roi_separation/np.max(roi_separation), roi_force, c="gray", alpha=0.5, linewidth=5)
    # plt.scatter(roi_separation_spline, roi_force_spline, c='r', s=2)
    # plt.show()
    curves_array[i] = roi_force_spline
    curves_array_xaxis[i] = np.linspace(min(roi_separation), max(roi_separation), 505)

# Storing the dataframes containing x and y axes of the force curves, for DTW comment the dropna() steps
df = pd.DataFrame(curves_array).T
df_xaxis = pd.DataFrame(curves_array_xaxis).T
df.dropna(axis=1, inplace=True)
df_xaxis.dropna(axis=1, inplace=True)
df.to_csv("ROI_ForceCurvesSpline.csv", sep=',')
df_xaxis.to_csv("ROI_ForceCurvesSpline_xaxis.csv", sep=',')
print(df)


    # plt.plot(roi_separation/np.max(roi_separation), roi_force, c="gray", alpha=0.5, linewidth=5)
    # plt.plot(roi_separation_spline, roi_force_spline, c='r', linewidth=1, linestyle='--')
    # plt.show()
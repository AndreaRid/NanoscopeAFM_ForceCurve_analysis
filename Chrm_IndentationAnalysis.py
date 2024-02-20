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
for curve in all_files:
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

    # Calculate the area under the indenting curve (i.e., energy of indentation)
    ind_area = simps(roi_force, dx=np.abs(roi_separation[1] - roi_separation[0]))
    ind_areas.append(ind_area)
    # Using spline interpolation to obtain a smoother trace
    spline_interp = UnivariateSpline(roi_separation, roi_force, k=1, s=55)
    # Generating finer x values for smooth plot
    roi_separation_spline = np.linspace(min(roi_separation), max(roi_separation), 500)
    # Interpolating y values using the spline interpolation function
    roi_force_spline = spline_interp(roi_separation_spline)
    # calculating the derivative of the indentation force
    df_dx = np.gradient(roi_force_spline)
    # denoinsing with Savitsky-Golay filter
    df_dx = savgol_filter(df_dx, 205, 3, mode='nearest')
    # storing useful descriptors for future clustering analysis
    df_in = pd.DataFrame({"curve_id": [curve],
                          "min_df_dx": [min(df_dx / df_dx[0])],
                          "max_df_dx": [max(df_dx / df_dx[0])],
                          "max_force": [max(roi_force)],
                          "cp_x": [cp_x],
                          "ind_energy": [ind_area / cp_x]})
    df = pd.concat([df, df_in])

    # Plotting the curve and its first derivative

    # color code for identification of curves with different features
    # if min(df_dx / df_dx[0]) <= .5 and max(roi_force) > 350:
    #     c = 'blue'
    #     df_in["color"] = ["blue"]
    # else:
    #     c = 'gray'
    #     df_in["color"] = ["gray"]
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(roi_separation, roi_force, alpha=0.3, c=c)
    # ax[0].scatter(roi_separation_spline, roi_force_spline, s=4)
    # ax[0].set_xlabel("Indentation (nm)")
    # ax[0].set_ylabel("Force (pN)")
    # ax[0].set_yscale("log")
    # ax[0].set_xscale("log")
    # ax[1].scatter(roi_separation_spline, df_dx, s=4, alpha=0.3, c=c)
    # ax[1].set_xlabel("Indentation (nm)")
    # ax[1].set_ylabel("dF/dx")
    # plt.show()

# De-comment the next line to save the features of all the processed curves into a .csv file for further analyses
# df.to_csv("PCA_dataset.csv")
# plt.savefig("Fig.1.png", dpi=200)

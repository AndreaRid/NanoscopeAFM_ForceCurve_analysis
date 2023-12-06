import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter  # Savitzky Golay filter for smoothing data with too much noise
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps   # Calculating the area under the curve
from sklearn.decomposition import PCA

# function for linear fit
def fit_func_linear(x, a, b):
    return a * x + b


# folders = ["Chromosomes/PS150452", "Chromosomes/Chromosome_PS161143"]
folders = [f for f in glob.glob("Chromosomes" + '**/*', recursive=True)]
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
spring_vals = []
# fig, ax = plt.subplots(1, 2)
df = pd.DataFrame()
all_curves = []
for curve in all_files:
    # name of the curve
    curve_name = curve[-11:]
    # fig, ax = plt.subplots(1, 2)
    # plt.ion()
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
    cp_x_values.append(cp_x)
    force = force - cp_y
    all_curves.append(curve)
    # Fitting: data preparation
    # should be zero at the contact pt., i.e. when the fit starts) setting the contact point ascissa as zero and
    # fitting a portion of curve n times the total indentation (cp_x - n * cp_x)
    n = .85
    # index of last point of the fitted section
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - n * cp_x]
    # portion of curve to be fitted
    fit_separation = separation[cp_index: end_pt[0]] - cp_x
    fit_force = force[cp_index: end_pt[0]]
    fit_separation = fit_separation * -1
    fit_separation = sorted(fit_separation)
    # Calculate the area under the indenting curve
    ind_area = simps(fit_force, dx=np.abs(fit_separation[1] - fit_separation[0]))
    ind_areas.append(ind_area)
    # Using spline interpolation to obtain a smoother trace
    spline_interp = UnivariateSpline(fit_separation, fit_force, k=1, s=55)
    # Generating finer x values for smooth plot
    fit_separation_spline = np.linspace(min(fit_separation), max(fit_separation), 500)
    # Interpolating y values using the spline interpolation function
    fit_force_spline = spline_interp(fit_separation_spline)
    df_dx = np.gradient(fit_force_spline)
    df_dx = savgol_filter(df_dx, 205, 3, mode='nearest')
    df_in = pd.DataFrame({"curve_id": [curve_name],
                          "min_df_dx": [min(df_dx / df_dx[0])],
                          "max_df_dx": [max(df_dx / df_dx[0])],
                          "max_force": [max(fit_force)],
                          "cp_x": [cp_x],
                          "ind_energy": [ind_area / cp_x]})
    if min(df_dx / df_dx[0]) <= .5 and max(fit_force) > 350:
        c = 'blue'
        # df_in["color"] = ["blue"]
    else:
        c = 'gray'
        # df_in["color"] = ["gray"]
    df = pd.concat([df, df_in])
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(fit_separation, fit_force, alpha=0.5, c=c)
    # ax[0].scatter(fit_separation_spline, fit_force_spline, s=4)
    # ax[0].set_yscale("log")
    # ax[0].set_xscale("log")
    # ax[1].scatter(fit_separation_spline, df_dx, s=4, alpha=0.5, c=c)
    # plt.show()

# plt.savefig("Fig.1.tif", dpi=600)
# plt.show()


# fig, ax = plt.subplots(1, 2)
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# sns.histplot(x=spring_vals, ax=ax[0])
# sns.scatterplot(x=cp_x_values, y=spring_vals, ax=ax[1])
# plt.show()



# Initialize PCA with desired number of components
pca = PCA(n_components=2)  # You can set the number of components

# Fit PCA on the data
data = df.drop("curve_id", axis=1)
pca.fit(data)

# Transform the data into principal components
transformed_data = pca.transform(data)

fig = plt.figure()
ax = fig.subplots(1, 2)


for i, curve in enumerate(all_curves[:200]):
    curve_data = nanoscope_converter(curve)
    separation = curve_data[6]
    force = curve_data[7]
    if transformed_data[i, 0] > 3000:
        c = 'red'
    else:
        c = 'blue'
    ax[0].plot(separation, force, alpha=.5, c=c)

# Plotting
# fig = plt.figure()
# ax = fig.subplots()
# ax = plt.axes(projection="3d")
# Transformed data (Principal Components)
# ax[1].scatter(transformed_data[:, 0], transformed_data[:, 1], c='red')
ax[1].scatter(transformed_data[:, 0], transformed_data[:, 1], c='blue')
# plt.title('Transformed Data (Principal Components)')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

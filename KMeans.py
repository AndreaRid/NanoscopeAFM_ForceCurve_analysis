import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer


'''This script imports the results from the Chrm_IndentationAnalysis.py and runs a KMeans clustering algorithm 
to understand if it's possible to cluster the indentation curves in distinct categories.'''


def fit_func_linear(x, a, b):
    '''function for linear fit.'''
    return a * x + b


# Initialize PCA
model = KMeans(n_clusters=3)
# Fit KMeans on the data
df = pd.read_csv("PCA_dataset.csv")
data = df.drop("curve_id", axis=1)
model.fit(data)
# Get cluster assignments for each data point
labels = model.labels_
# Get cluster centers
# centers = model.cluster_centers_

# print("labels", labels)
# print("centers", centers)


def fit_func_linear(x, a, b):
    '''function for linear fit.'''
    return a * x + b

# Plotting
fig = plt.figure(figsize=(20, 10))
ax = fig.subplots(1, 2)
# First we need to process and plot all the raw curves data
for i, curve in enumerate(df["curve_id"].values):
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
    # Selecting only a portion of the indentation curve, n times the total indentation (cp_x - n * cp_x)
    n = .85
    # index of last point of the selected portion
    end_pt = [i for i in range(len(separation)) if separation[i] < cp_x - n * cp_x]
    # portion of curve to be displayed
    fit_separation = separation[cp_index: end_pt[0]] - cp_x
    fit_force = force[cp_index: end_pt[0]]
    fit_separation = fit_separation * -1
    color = ["gray", "blue", "darkred"]
    ax[0].plot(fit_separation, fit_force, alpha=0.3, c=color[labels[i]])
    ax[0].set_xlabel("Indentation (nm)")
    ax[0].set_ylabel("Force (pN)")

plt.tight_layout()
# saving the plot
plt.savefig("KMeans_clusters.png", dpi=300)
plt.show()

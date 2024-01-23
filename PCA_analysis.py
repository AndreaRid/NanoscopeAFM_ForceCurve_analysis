import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer


'''This script imports the results from the Chrm_IndentationAnalysis.py and runs a Principal Component Analysis (PCA) 
to understand which are the main properties that can be used to cluster different types of curves.'''


def fit_func_linear(x, a, b):
    '''function for linear fit.'''
    return a * x + b


# Initialize PCA
pca = PCA(n_components=2)
# Fit PCA on the data
df = pd.read_csv("PCA_dataset.csv")
data = df.drop("curve_id", axis=1)
pca.fit(data)
# Transform the data into principal components
transformed_data = pca.transform(data)
# Number of principal components
n_pcs = pca.components_.shape[0]
print("Number of Principal Components: ", n_pcs)
print("Value of the Principal Components: ", pca.components_)
# get the index of the most important feature on each component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = data.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print("Most important Principal Components: ", most_important_names)


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

    # After checking PCA results, manually group the points in the 2 PC plot into 3 distinct clusters
    # selecting points in the central part of the cluster
    if (transformed_data[i, 0] > -3820) and (transformed_data[i, 0] < 3360) and (transformed_data[i, 1] > -48):
        c = 'blue'
    # selecting points in the negative slope cluster
    elif not((transformed_data[i, 0] > -3820) and (transformed_data[i, 0] < 3360) and (transformed_data[i, 1] > -48)) and transformed_data[i, 0] < 3000:
        c = 'gray'
    # selecting points in the positive slope cluster
    elif not((transformed_data[i, 0] > -3820) and (transformed_data[i, 0] < 3360) and (transformed_data[i, 1] > -48)) and transformed_data[i, 0] > 3000:
        c = 'darkred'
    ax[0].plot(fit_separation, fit_force, alpha=0.3, c=c)
    ax[0].set_xlabel("Indentation (nm)")
    ax[0].set_ylabel("Force (pN)")
    ax[1].scatter(transformed_data[i, 0], transformed_data[i, 1], c=c)
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')

plt.tight_layout()
# saving the plot
plt.savefig("PCA_clusters.png", dpi=300)
plt.show()

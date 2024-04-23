import pandas as pd
import numpy as np
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurvesContactPoint_determination import contact_pointFinder2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

'''This script imports the results from the PCA_KMeans_DataPrep.py and runs a Principal Component Analysis (PCA) 
to understand which are the main properties that can be used to cluster different types of curves.'''


def fit_func_linear(x, a, b):
    '''function for linear fit.'''
    return a * x + b


# Loading the curve dataset
curr_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.dirname(curr_dir) + "/CurveFeatures_dataset.csv"
df = pd.read_csv(filepath).iloc[:, 1:]
# for running PCA and other analysis, the curve ID is not useful
data = df.drop("curve_id", axis=1)
# Scaling data
scaler = StandardScaler()
# scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# Plotting original and scaled dataset
fig, ax = plt.subplots(1, 2)
for i in range(data.shape[1]):
    sns.kdeplot(data.iloc[:, i], label=data.columns[i], ax=ax[0])
    ax[0].set_ylim(0, 0.05)
    ax[0].set_title("Original values")
    sns.kdeplot(scaled_df.iloc[:, i], label=data.columns[i], ax=ax[1])
    ax[1].set_title("Scaled values")
plt.legend()
plt.show()

# Reduce to two main principal components
pca = PCA(n_components=2, random_state=21)
pca.fit(scaled_df)
transformed_data = pca.transform(scaled_df)
# Number of principal components
n_pcs = pca.components_.shape[0]
# print("Number of Principal Components: ", n_pcs)
print("Value of the Principal Components: ", pca.components_)
# get the index of the most important feature on each component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = data.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print("Most important Principal Components: ", most_important_names)
fig = plt.figure()
ax = fig.subplots()
# Plotting
ax.scatter(transformed_data[:, 0], transformed_data[:, 1])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.show()


# Using PCA for understanding which components can explain the variance of the dataset
pca = PCA(random_state=21)
pca.fit(scaled_df)
expl_var_ratio = pca.explained_variance_ratio_
print(pd.DataFrame({"Component": data.columns,
                   "Variance ratio of the components":expl_var_ratio}))
# How many features to explain 90% of variance
fig, ax = plt.subplots()
plt.title('How many features to explain 90% of variance?')
ax.plot(range(1, data.shape[1]+1), expl_var_ratio.cumsum(), marker='o')
ax.axhline(y=0.9, color='g', linestyle='--')
ax.set_ylim(0.3, 1.2)
plt.savefig("KMean_after_PCA_VarianceRatio.png", dpi=600)
plt.show()

# Using 4 components we can explain > 90% variance
pca = PCA(n_components=4, random_state=21)
pca_df = pd.DataFrame(pca.fit_transform(scaled_df))
# to figure out the features that are important
sd = np.abs(pca.components_[0]).argsort()[::-1]
major_components = pd.DataFrame(np.abs(pca.components_[0]),
                                index=scaled_df.columns[sd],
                                columns=['Value'])
print("Major components: ")
print(major_components.sort_values(by='Value', ascending=False))
# Elbow method for deciding the optimal number of clusters in KMeans
inertia = []

for i in range(1, pca_df.shape[1]+1):
    # performing KMeans for different cluster numbers and calculating the inertia
    KM = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=21)
    KM.fit(pca_df)
    inertia.append(KM.inertia_)
plt.title('KMeans of the PCA')
plt.plot(range(1, pca_df.shape[1]+1), inertia, marker='o')
plt.xlabel('Clusters')
plt.ylabel('KMeans Inertia')
plt.savefig("KMean_after_PCA_Elbow.png", dpi=600)
plt.show()

# After deciding the optimal number of clusters, perform KMeans
Kmeans_pca = KMeans(n_clusters=3, init='k-means++', n_init='auto', random_state=21)
Kmeans_pca.fit(pca_df)
# these are the cluster labels
cluster_labels = Kmeans_pca.labels_
# creating a dataframe from the components used by the KMeans
df_pc = pd.DataFrame(pca_df)
df_pc['Label'] = cluster_labels
print(df_pc)
# plotting the clusters
sns.scatterplot(data=df_pc, x=df_pc.iloc[:,0], y=df_pc.iloc[:,1], hue='Label', palette="viridis")
plt.title('Scatter plot of the clusters')
plt.scatter(Kmeans_pca.cluster_centers_[:, 0], Kmeans_pca.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.savefig("KMean_after_PCA_Clusters&Centroids.png", dpi=300)
plt.show()

# Check which pair of variables is better to visualize the clusters in 2D
# sns.pairplot(df_pc, hue="Label")
# plt.show()

fig, ax = plt.subplots(1, 2)
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
    # if cp_x <= 150:
    #     continue
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
    cluster_n = df_pc.iloc[i, -1]
    c = ["blue", "gray", "darkred", "green"]
    ax[0].plot(fit_separation, fit_force, alpha=0.3, c=c[cluster_n])
    ax[0].set_xlabel("Indentation (nm)")
    ax[0].set_ylabel("Force (pN)")
    ax[1].scatter(df_pc.iloc[i, 0], df_pc.iloc[i, 1], c=c[cluster_n])
    ax[1].set_xlabel(df_pc.columns[0])
    ax[1].set_ylabel(df_pc.columns[1])
plt.tight_layout()
# saving the plot
plt.savefig("KMean_after_PCA_Clusters&Curves.png", dpi=600)
plt.show()

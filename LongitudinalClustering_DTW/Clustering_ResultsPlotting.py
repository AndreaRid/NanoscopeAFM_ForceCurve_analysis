import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# dataframes for plotting results from DTW
df = pd.read_csv("ROI_ForceCurvesSpline_DTW.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis_DTW.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1)


def plot_dendrogram(Z):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(15, 5))
         plt.title('Dendrogram for curve clustering', fontsize=25, fontweight='bold')
         plt.xlabel('sample index', fontsize=15)
         plt.ylabel('distance', fontsize=15)
         dendrogram(Z, leaf_rotation=90.,  # rotates the x axis labels
                        leaf_font_size=15) # and font size for the x axis labels
         plt.show()


def plot_results(yaxis, xaxis, Z, cut_off_level):
    # assign each curve to a specific cluster (based on cut_off_level)
    result = pd.Series(fcluster(Z, cut_off_level, criterion='distance'))
    clusters = result.unique()
    fig, ax1 = plt.subplots(nrows=1, ncols=len(clusters))
    fig, ax2 = plt.subplots()
    palette = sns.color_palette("magma_r", len(clusters))
    # parameter that regulates the zorder for the plots, brighter on upper levels
    l = len(clusters)
    for i, c in enumerate(sorted(clusters)):
        # c is the number of the cluster under evaluation, index of results contains the curve numbers
        cluster_index = result[result==c].index.astype(float)
        print(i, "Cluster number %d has %d elements" % (c, len(cluster_index)))
        # plotting all the curves whose name (number) is in cluster_index
        ax1[i].plot(xaxis.iloc[:, cluster_index].dropna(axis=1),
                    yaxis.iloc[:, cluster_index].dropna(axis=1), alpha=0.5)
        ax1[i].set_title(('Cluster number '+str(c)), fontsize=15, fontweight='bold')
        ax1[i].set_ylim(-200, 12100)
        ax1[i].set_xlim(0, 550)
        # plotting all the curves in the same plot differentiated by color
        ax2.plot(xaxis.iloc[:, cluster_index].dropna(axis=1), yaxis.iloc[:, cluster_index].dropna(axis=1),
                 c=palette[i], alpha=0.5, zorder=l)
        ax2.set_ylim(-200, 12100)
        ax2.set_xlim(0, 550)
        l -= 1
    plt.show()


# importing the distance matrix
condensed_distances = np.loadtxt('condensed_distances_parallel.csv', delimiter=',')

# # Perform hierarchical clustering
Z = linkage(condensed_distances, method='ward', metric='euclidean')  # You can choose different linkage methods
plot_dendrogram(Z)
plt.show()

#---- evaluate the dendrogram (input the value of "distance" where you want to cut the dendrogram)
cut_off_level = input("insert cut off level: ")
plot_results(df, df_xaxis, Z, cut_off_level)

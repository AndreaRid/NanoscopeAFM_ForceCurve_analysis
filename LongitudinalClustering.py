import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import fastdtw
from scipy.spatial.distance import squareform
import time



def plot_dendrogram(Z):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(15, 5))
         plt.title('Dendrogram for curve clustering', fontsize=25, fontweight='bold')
         plt.xlabel('sample index', fontsize=15)
         plt.ylabel('distance', fontsize=15)
         dendrogram(Z, leaf_rotation=90.,  # rotates the x axis labels
                        leaf_font_size=15) # and font size for the x axis labels
         plt.show()

def plot_results(yaxis, xaxis, D, cut_off_level):
    # assign each curve to a specific cluster (based on cut_off_level)
    result = pd.Series(fcluster(D, cut_off_level, criterion='distance'))
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



'''Using Euclidean Distance as metrics'''
df = pd.read_csv("ROI_ForceCurvesSpline.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1)

# run the clustering
#D = linkage(df.T, method='single', metric='correlation')
# D = linkage(df, method='ward', metric='euclidean')
# print("Linkage matrix: ", D)
# plot_dendrogram(D)

#---- evaluate the dendrogram
# cut_off_level = .5e6# level where to cut off the dendrogram
# plot_results(df.T, df_xaxis.T, D, cut_off_level)


'''Dynamic Time Warping'''

df = pd.read_csv("ROI_ForceCurvesSpline_DTW.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis_DTW.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1)
# Compute pairwise DTW distances
num_curves = df.shape[1]
dtw_distances = np.zeros((num_curves, num_curves))


# print(dtw_distances)
start_time = time.time()
for i in range(num_curves):
    x = df.iloc[:, i].dropna().values
    for j in range(i + 1, num_curves):
        y = df.iloc[:, j].dropna().values
        dtw_distances[i, j] = fastdtw.fastdtw(x, y)[0]
        dtw_distances[j, i] = dtw_distances[i, j]  # Distance matrix is symmetric
end_time = time.time()
print('Processing time: ', end_time - start_time, " sec")
condensed_distances = squareform(dtw_distances)

# Saving the condensed distance matrix as .csv for future use
np.savetxt('condensed_distances.csv', condensed_distances, delimiter=',')




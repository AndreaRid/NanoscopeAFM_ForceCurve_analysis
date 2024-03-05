import numpy as np
import pandas as pd
import fastdtw # for Dynamic Time Warping
from scipy.spatial.distance import squareform
import time

'''Script for calculating distance matrix between different curves using different Longitudinal 
clustering approaches.'''


'''Using Euclidean Distance as metrics'''
# dataframe containing the force (y-values) of the curves
df = pd.read_csv("ROI_ForceCurvesSpline.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
# dataframe containing the separation (x-values) of the curves
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1)
# run the clustering
# D = linkage(df.T, method='ward', metric='euclidean')
# print("Linkage matrix: ", D)

'''Dynamic Time Warping (DTW) --- better results ----'''
# dataframe containing the force (y-values) of the curves
df = pd.read_csv("ROI_ForceCurvesSpline_DTW.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
# dataframe containing the separation (x-values) of the curves
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis_DTW.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1)
# initialize distance matrix
num_curves = df.shape[1]
dtw_distances = np.zeros((num_curves, num_curves))
# print(dtw_distances)
start_time = time.time()
# Compute pairwise DTW distances
for i in range(num_curves):
    x = df.iloc[:, i].dropna().values
    for j in range(i + 1, num_curves):
        y = df.iloc[:, j].dropna().values
        dtw_distances[i, j] = fastdtw.fastdtw(x, y)[0]
        dtw_distances[j, i] = dtw_distances[i, j]  # Distance matrix is symmetric
end_time = time.time()
print('Processing time: ', end_time - start_time, " sec")
# transforming the distance matrix into its condensed form for computing the dendrogram
condensed_distances = squareform(dtw_distances)
# Saving the condensed distance matrix as .csv for future use
np.savetxt('condensed_distances.csv', condensed_distances, delimiter=',')
import pandas as pd
from multiprocessing import Pool
from scipy.spatial.distance import squareform
import numpy as np
import fastdtw
import time

def calculate_dtw_distance(indexes):
    i, j = indexes[0], indexes[1]
    df = indexes[2]
    x = df.iloc[:, i].dropna().values
    y = df.iloc[:, j].dropna().values
    return fastdtw.fastdtw(x, y)[0], i, j



'''Dynamic Time Warping'''

# print("Starting")
# Create a pool of worker processes
if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv("ROI_ForceCurvesSpline_DTW.csv", sep=',')
    df = df.drop('Unnamed: 0', axis=1)
    # Compute pairwise DTW distances with parallelization
    num_curves = df.shape[1]
    with Pool() as pool:
        # Parallelize the computation of DTW distances
        iterators = []
        for i in range(num_curves):
            for j in range(i + 1, num_curves):
                iterators.append([i, j, df])
        results = pool.map(calculate_dtw_distance, iterators)
    end = time.time()
    print("Processing time: ", end-start, " sec")
    # Initialize the distance matrix
    dtw_distances = np.zeros((num_curves, num_curves))
    # Update the distance matrix with the results
    for distance, i, j in results:
        dtw_distances[i, j] = distance
        dtw_distances[j, i] = distance  # Distance matrix is symmetric

    condensed_distances = squareform(dtw_distances)
    # Saving the condensed distance matrix as .csv for future use
    np.savetxt('condensed_distances_parallel.csv', condensed_distances, delimiter=',')



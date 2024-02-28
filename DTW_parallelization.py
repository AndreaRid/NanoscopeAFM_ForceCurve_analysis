import pandas as pd
from multiprocessing import Pool
from scipy.spatial.distance import squareform
import numpy as np
import fastdtw
from time import time, sleep
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def calculate_dtw_distance(indexes):
    '''Calculate the distance between the i-th and j-th columns (curves) of the dataframe by using Dynamic Time
    Warping. Returns the distance and the indexes of the respective columns.'''
    i, j, df = indexes[0], indexes[1], indexes[2]
    x = df.iloc[:, i].dropna().values
    y = df.iloc[:, j].dropna().values
    return fastdtw.fastdtw(x, y)[0], i, j



# Performing DTW analysis exploiting parallelization
if __name__ == '__main__':
    start = time()
    df = pd.read_csv("ROI_ForceCurvesSpline_DTW.csv", sep=',')
    df = df.drop('Unnamed: 0', axis=1)
    # Compute pairwise DTW distances with parallelization
    num_curves = df.shape[1]
    # generating pool of processes
    with Pool() as pool:
        iterators = []
        # pairwise distance calculation using fastdtw
        for i in range(num_curves):
            for j in range(i + 1, num_curves):
                iterators.append([i, j, df])
        # storing both the distance and the indexes for creating distance matrix once finished
        results = pool.map(calculate_dtw_distance, iterators)
    end = time()
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



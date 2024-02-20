import math
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_data(nT,nC,mG,A,sg,eg):
    timeSeries = pd.DataFrame()
    basicSeries = pd.DataFrame()
    β = 0.5*np.pi
    ω = 2*np.pi/nT
    t = np.linspace(0,nT,nT)
    for ic,c in enumerate(np.arange(nC)):
        slope = sg*(-(nC-1)/2 + c)
        s = A * (-1**c -np.exp(t*eg))*np.sin(t*ω*(c+1) + c*β) + t*ω*slope
        basicSeries[ic] = s
        sr = np.outer(np.ones_like(mG),s)
        sr = sr + 1*np.random.rand(mG,nT) + 1.0*np.random.randn(mG,1)
        timeSeries = timeSeries.append(pd.DataFrame(sr))
    return basicSeries, timeSeries

def plot_basicSeries(basicSeries):
    with plt.style.context('seaborn'):      # 'fivethirtyeight'
         fig = plt.figure(figsize=(20,8)) ;
         ax1 = fig.add_subplot(111);
         plt.title('Basice patterns to generate Longitudinal data',fontsize=25, fontweight='bold')
         plt.xlabel('Time', fontsize=15, fontweight='bold')
         plt.ylabel('Signal of the observed feature', fontsize=15, fontweight='bold')
         plt.plot(basicSeries, lw=10, alpha=.8)

def plot_timeSeries(timeSeries):
    with plt.style.context('seaborn'):      # 'fivethirtyeight'
         fig = plt.figure(figsize=(20,8)) ;
         ax1 = fig.add_subplot(111);
         plt.title('Longitudinal data',fontsize=25, fontweight='bold')
         plt.xlabel('Time', fontsize=15, fontweight='bold')
         plt.ylabel('Signal of the observed feature', fontsize=15, fontweight='bold')
         plt.plot(timeSeries.T)
         #ax1 = sns.tsplot(ax=ax1, data=timeSeries.values, ci=[68, 95])

def plot_dendogram(Z):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(15, 5))
         plt.title('Dendrogram of time series clustering',fontsize=25, fontweight='bold')
         plt.xlabel('sample index', fontsize=25, fontweight='bold')
         plt.ylabel('distance', fontsize=25, fontweight='bold')
         hac.dendrogram(Z, leaf_rotation=90.,  # rotates the x axis labels
                        leaf_font_size=15) # and font size for the x axis labels
         plt.show()

def plot_results(timeSeries, xaxis, D, cut_off_level):
    result = pd.Series(hac.fcluster(D, cut_off_level, criterion='distance'))
    clusters = result.unique()
    print(result.value_counts())
    fig, ax1 = plt.subplots(nrows=1, ncols=len(clusters))
    fig, ax2 = plt.subplots()
    palette = sns.color_palette("magma", len(clusters))
    # palette = ['gray', 'darkred', 'blue', 'darkred', 'gray']
    l = len(clusters)
    for i, c in enumerate(sorted(clusters)):
        cluster_index = result[result==c].index.astype(float)
        # print(cluster_index)
        print(i, "Cluster number %d has %d elements" % (c, len(cluster_index)))
        # print(type(cluster_index))
        # print(xaxis.iloc[:, cluster_index].columns)
        # print(timeSeries.iloc[:, cluster_index].columns)
        ax1[i].plot(xaxis.iloc[:, cluster_index], timeSeries.iloc[:, cluster_index], alpha=0.5)
        ax1[i].set_title(('Cluster number '+str(c)), fontsize=15, fontweight='bold')
        ax1[i].set_ylim(-200, 12100)
        ax1[i].set_xlim(0, 550)
        ax2.plot(xaxis.iloc[:, cluster_index], timeSeries.iloc[:, cluster_index],
                 c=palette[i], alpha=0.5, zorder=l)
        ax2.set_ylim(-200, 12100)
        ax2.set_xlim(0, 550)
        l -= 1
    plt.show()

def plot_basic_cluster(X):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(17,3))
         D1 = hac.linkage(X, method='ward', metric='euclidean')
         dn1= hac.dendrogram(D1)
         plt.title("Clustering: method='ward', metric='euclidean'")

         plt.figure(figsize=(17, 3))
         D2 = hac.linkage(X, method='single', metric='euclidean')
         dn2= hac.dendrogram(D2)
         plt.title("Clustering: method='single', metric='euclidean'")
         plt.show()



'''Generating Data'''
#---- number of time series
# nT = 101  # number of observational point in a time series
# nC = 6    # number of charakteristic  signal groups
# mG = 10   # number of time series in a charakteristic signal group

#---- control parameters for data generation
# Am = 0.3; # amplitude of the signal
# sg = 0.3  # rel. weight of the slope
# eg = 0.02 # rel. weight of the damping

#---- generate the data
# basicSeries,timeSeries = generate_data(nT,nC,mG,Am,sg,eg)
# plot_basicSeries(basicSeries)
# plot_timeSeries(timeSeries)


df = pd.read_csv("ROI_ForceCurvesSpline.csv", sep=',')
df = df.drop('Unnamed: 0', axis=1)
df_xaxis = pd.read_csv("ROI_ForceCurvesSpline_xaxis.csv", sep=',')
df_xaxis = df_xaxis.drop('Unnamed: 0', axis=1).T
print("DF", df.columns)
print("X_axis", df_xaxis.columns)
# fig, ax = plt.subplots()
# for i in range(df.shape[1]):
#     ax.plot(df.index, df.iloc[:, i])
# plt.show()

'''Using Euclidean Distance as metrics'''
#--- run the clustering
#D = hac.linkage(timeSeries, method='single', metric='correlation')
# D = hac.linkage(df, method='ward', metric='euclidean')
# print("Linkage matrix: ", D)
# plot_dendogram(D)

#---- evaluate the dendogram
# cut_off_level = .5e6# level where to cut off the dendogram
# plot_results(df.T, df_xaxis.T, D, cut_off_level)

from tslearn.metrics import dtw
from scipy.cluster.hierarchy import dendrogram, linkage

# alignment = dtw(df.iloc[:, 0], df.iloc[:, 506], keep_internals=True)
# print(alignment)
num_time_series = df.shape[1]
dtw_distances = np.zeros((num_time_series, num_time_series))
for i in range(num_time_series):
    for j in range(i + 1, num_time_series):
        dtw_distances[i, j] = dtw(df.iloc[:, i], df.iloc[:, j])
        dtw_distances[j, i] = dtw_distances[i, j]  # Distance matrix is symmetric

# Perform hierarchical clustering
Z = linkage(dtw_distances, method='average')
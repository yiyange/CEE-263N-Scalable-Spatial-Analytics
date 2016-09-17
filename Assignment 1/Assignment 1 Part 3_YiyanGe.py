
# coding: utf-8

# Part 3. Clustering: 1 million samples problem (40 points). This part deals with the full dataset of 1 million tweets. Your task is to design a system that can handle spatial clustering of 1M samples.
# Considering the memory limitations and scaling properties of the algorithms studied in Part 2, design the clustering system that can be applied to the full dataset. Consider using a hierarchical approach with two (or more) processing stages, where DBScan is applied to each cluster obtained from a run of mini-batch k-means. By varying the parameters of the algorithms, optimize the processing time required to detect clusters of tweets that correspond to important locations in California. We will consider a location “important” if it is characterized with a cluster’s core of at least 100 samples within a radius of 100 meters.

# In[1]:

import time
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import math
import json
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')


# In[2]:

with open('data/tweets_1M.json','r') as f:
    tweets = json.load(f)


# In[3]:

X = np.array([[tweets[x]['lat'],tweets[x]['lng']] for x in range(0, len(tweets))])


# In[4]:

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


# In[5]:

import utm
for n in range(0, len(X)):
    meters = utm.from_latlon(X[n][0],X[n][1])
    X[n][0] = meters[0]
    X[n][1] = meters[1]


# In[13]:

#corresponds to step 3 in the report
for perc in [0.001,0.002,0.003, 0.004]:
    
    for n in range(60, 71, 1):
    
        batch_size=int(len(X)*perc)
        ttl_t = 0

        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(X)
        t_mini_batch = time.time() - t0
        ttl_t += t_mini_batch

        mbk_means_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(mbk_means_labels)

        for cluster_label in list(mbk_means_labels_unique):

            mask = (mbk_means_labels == cluster_label)
            X_cluster = X[mask]

            eps = 100
            t_db = time.time()
            db = DBSCAN(eps=eps, min_samples=100).fit(X_cluster)
            t_fin_db = time.time() - t_db

            ttl_t += t_fin_db

        print (perc*1000000, n, ttl_t)


# In[ ]:




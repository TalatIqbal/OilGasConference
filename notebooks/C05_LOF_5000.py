
# coding: utf-8

# ## Local Outlier Factor - Neighbours

# In[18]:


import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from scipy.spatial import distance_matrix
from sklearn.neighbors import LocalOutlierFactor

def reach_dist(idx_o, idx_o_dash, X_dist, knn_dist):
    return max(X_dist[idx_o][idx_o_dash], max(knn_dist[idx_o]))


# ### Find the Local Reachability Distance and Local Outlier Factor

# In[21]:


def get_lrd_lof(k, X, X_dist):
    
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    pred_labels = clf.fit_predict(X)
    knn_dist, knn_indices = clf.kneighbors(X=X, n_neighbors=k, return_distance=True)

    lrd = list()
    for element_idx in range(0, len(knn_indices)):
        knn_elements_indices = knn_indices[element_idx]
        lrd_val = 0
        for neighbour_idx in knn_elements_indices:
            lrd_val += reach_dist(neighbour_idx, element_idx, X_dist, knn_dist)
        lrd_val /= k
        lrd.append(lrd_val)

    lof = list()
    for element_idx in range(0, len(knn_indices)):
        knn_elements_indices = knn_indices[element_idx]
        lof_val = 0
        for neighbour_idx in knn_elements_indices:
            lof_val += lrd[neighbour_idx]

        lof_val /= lrd[element_idx]
        lof_val /= k

        lof.append(lof_val)
        
    return lof, lrd


def main():

    INPUT_DIR = '../data/raw/CMAPSSData/'
    OUTPUT_DIR = '../data/interim/'

    input_file = INPUT_DIR + 'train_FD003.txt'
    col_headers = ['unit', 'time_cycles', 'setting1', 'setting2', 'setting3', 
                  'meas01', 'meas02', 'meas03', 'meas04', 'meas05', 'meas06', 'meas07', 'meas08', 'meas09', 'meas10', 
                  'meas11', 'meas12', 'meas13', 'meas14', 'meas15', 'meas16', 'meas17', 'meas18', 'meas19', 'meas20', 
                  'meas21', 'meas22', 'meas23', 'meas24', 'meas25', 'meas26']

    df = pd.read_csv(input_file, header=None, sep=' ', names=col_headers)

    id_cols = ['unit', 'time_cycles']
    feature_set = ['setting1', 'setting2', 'setting3', 
                  'meas01', 'meas02', 'meas03', 'meas04', 'meas05', 'meas06', 'meas07', 'meas08', 'meas09', 'meas10', 
                  'meas11', 'meas12', 'meas13', 'meas14', 'meas15', 'meas16', 'meas17', 'meas20', 
                  'meas21']

    cols = feature_set + id_cols

    df = df[cols]

    X = df[feature_set]

    normalized_X = preprocessing.normalize(X)

    X_dist = distance_matrix(X, X)

    print('X_dist computed')

    k_list = [5000]
    print('Preparing list for k values ',  k_list)

    print('| k | Outliers | Time(s) |')
    print('| -- | -- | -- |')
    for k in k_list:
        tic = time.time()
        lof, lrd = get_lrd_lof(k, X, X_dist)

        df['lof'] = lof
        df['lrd'] = lrd

        op_fname = OUTPUT_DIR + 'fd001_lof_' + str(k) + '.csv'

        with open(op_fname, 'w') as f:
            df.to_csv(op_fname)

        outliers_count = len([i for i in lof if i > 1.0])
        print('|', k, '|', outliers_count, '|', (time.time()-tic), '|')


main()



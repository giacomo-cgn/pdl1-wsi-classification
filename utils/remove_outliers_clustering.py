import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.metrics import silhouette_samples


def distance_from_centroid(cluster_embeds, centroid):
    distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
    #print('distances arr len:', distances.size)
    return distances


def remove_outliers_clustering(clusters_max_dist_arr, all_labels, all_embeds, n_clusters, centroids):
    new_labels = np.copy(all_labels)
        
    for c in range(n_clusters):
       
        # Find all embeddings for a cluster
        cluster_embeds = [all_embeds[i] for i, x in enumerate(all_labels) if x == c]
        # Save original array index of the labels for a cluster
        cluster_indexes = [i for i, x in enumerate(all_labels) if x == c]
        
        centroid = centroids[c]
        cluster_max_dist = clusters_max_dist_arr[c]
        
        cluster_embeds = np.array(cluster_embeds)
        
        if cluster_embeds.shape[0] == 0:
            # No elements in that cluster
            distance_arr = []
        else:
            # Find arr of distances from centroid for a cluster
            distance_arr = distance_from_centroid(np.array(cluster_embeds), centroid)
        
        for i, distance in enumerate(distance_arr):
            if distance>cluster_max_dist:
                new_labels [cluster_indexes[i]] = -1

    return new_labels



def find_and_remove_outliers_clustering(all_labels, all_embeds, n_clusters, centroids, percentile):
    clusters_max_dist_arr = []
    new_labels = np.copy(all_labels)

    for c in range(n_clusters):
        # Find all embeddings for a cluster
        cluster_embeds = [all_embeds[i] for i, x in enumerate(all_labels) if x == c]
        # Save original array index of the labels for a cluster
        cluster_indexes = [i for i, x in enumerate(all_labels) if x == c]
        
        centroid = centroids[c]
        
        # Find arr of distances from centroid for a cluster
        distance_arr = distance_from_centroid(np.array(cluster_embeds), centroid)
        
        #Find specified percentile distance for a cluster (will be max non oultier distance)
        cluster_max_dist = np.percentile(a=distance_arr, q=percentile)
        clusters_max_dist_arr.append(cluster_max_dist)
        
        # Set to None all the labels over max distance for cluster
        for i, distance in enumerate(distance_arr):
            if distance>cluster_max_dist:
                new_labels [cluster_indexes[i]] = -1
        
        
    return clusters_max_dist_arr, new_labels
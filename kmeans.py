"""Package created by: Nicolás PÉREZ

Date: 17/11/2023

This package was developped as an exercice for the course: Algorithms for Data Analysis, of Télécom St. Étienne Engineering school

It implements the algorithm of KMeans and lets you plot a 2D or 3D representation of the clusters

--------------

Dependencies:

    numpy
    matplotlib
 """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random as rd

def distance(u: np.ndarray, v: np.ndarray) -> float:
    """Calculates the Euclidean distance between two vectors.

    Parameters
    ----------
    u, v : np.ndarray
        Array containing the Euclidean coordinates of a point.

    Returns
    -------
    float
        Number indicating the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((u - v) ** 2))

def kMeans(data: np.ndarray, k: int = 3, axis: int = 0) -> np.ndarray:
    """Calculates the clusters for a given dataset using the K-Means algorithm.

    Parameters
    ----------
    data : np.ndarray
        Array containing the coordinates of all the points on a 2-dimensional space.
    k : int, optional
        Number of clusters (default is 3).
    axis : int, optional
        Number defining according to which axis the coordinates are displayed, for 0 it would be a table with the first column corresponding to all
        the X-axis values, the second column corresponds to all Y-axis values, etc. (default is 0).

    Returns
    -------
    list
        A list of clusters indicating the cluster to which each element of the dataset belongs.
    """
    #Condition necessary in case the data is not in the right order
    if axis == 1:
        data = np.transpose(data)

    #Array containing the list of the cluster corresponding to every point on the dataset
    clusters = np.zeros(np.shape(data)[0])

    #Array used for the KMeans Algorithm cotaining the cluster centers
    centroids = []

    #Array used for verifying the convergence of the algorithm
    convergenceCentroids = []

    # Step 1: Random point choice
    for i in range(k):
        randomIdx = rd.randint(0, np.shape(data)[0])
        centroids.append(np.array([data[randomIdx]]))

    convergenceCentroids.append(centroids)

    #Loop charged of the KMeans algorithm 
    while True:
        # Step 2: finding the nearest centroid for each point x
        for i in range(np.shape(data)[0]):
            distances = []
            for c in range(k):
                vect = np.array([data[i]])
                dist = distance(centroids[c], vect)
                distances.append(dist)
            minDist = min(distances)
            minDistIdx = distances.index(minDist)
            clusters[i] = minDistIdx

        # Step 3: Define new centroids
        for i in range(k):
            associatedClusterXi = []

            for j in range(len(clusters)):
                if clusters[j] == i:
                    associatedClusterXi.append(np.array([data[j]]))

            if len(associatedClusterXi) > 0:
                centroids[i] = np.sum(associatedClusterXi, axis=0) / len(associatedClusterXi)

        #Loop breakpoint, when no more centroids are available, the algorithm is stopped
        if len(convergenceCentroids) < 2:
            convergenceCentroids.append(centroids)
        else:
            convergenceCentroids[0], convergenceCentroids[1] = convergenceCentroids[1], centroids

        convergenceDist = 0
        for i in range(k):
            convergenceDist += distance(convergenceCentroids[0][i], convergenceCentroids[1][i])

        if convergenceDist == 0:
            break

    return clusters

def plotKMeans(data: np.ndarray, k: int = 3, axis: int = 0, clusters: list = []):
    """Plots the results of the K-Means algorithm in a 2D or 3D espace.

    Parameters
    ----------
    data : np.ndarray
        Array containing the coordinates of all the points on a 2-dimensional space or 3-dimensional space.
    k : int, optional
        Number of clusters (default is 3).
    axis : int, optional
        Number defining according to which axis the coordinates are displayed, for 0 it would be a table with the first column corresponding to all
        the X-axis values, the second column corresponds to all Y-axis values, etc. (default is 0).
    clusters : list, optional
        A list of clusters indicating the cluster to which each element of the dataset belongs. If not provided, it will be calculated using the K-Means algorithm.

    Returns
    -------
    None
    """

    #Condition necessary in case the data is not in the right order
    if axis == 1:
        data = np.transpose(data)

    if len(clusters) == 0:
        clusters = kMeans(data, k)

    if(np.shape(data)[1] == 2):
        fig, ax = plt.subplots(1, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            group_x = []
            group_y = []
            for j in range(np.shape(data)[0]):
                if clusters[j] == i:
                    group_x.append(data[j, 0])
                    group_y.append(data[j, 1])
            ax.scatter(group_x, group_y, color=colors[i], s=3, label=f"Group {i+1}")

        ax.legend()

    elif np.shape(data)[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(k):
            group_x = []
            group_y = []
            group_z = []
            for j in range(np.shape(data)[0]):
                if clusters[j] == i:
                    group_x.append(data[j, 0])
                    group_y.append(data[j, 1])
                    group_z.append(data[j, 2])
            ax.scatter(group_x, group_y, group_z, s=3, label=f"Group {i+1}")

        ax.legend()

    else:
        raise ValueError("Data must have either 2 or 3 dimensions for plotting.")


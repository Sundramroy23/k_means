import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
X, Y = make_blobs(n_samples=500, n_features=2, centers=5, random_state=40)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=Y)
plt.grid(True)
plt.show()

# Define the number of clusters
k = 5
colors = ['green', 'yellow', 'blue', 'cyan', 'red']
clusters = {}

# Initialize random centers
for idx in range(k):
    center = 10 * (2 * np.random.random(X.shape[1]) - 1)
    cluster = {
        'center': center,
        'points': [],
        'color': colors[idx]
    }
    clusters[idx] = cluster

# Plot initial centers
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=Y)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='^', c="red")
plt.show()

# Function to find the distance between two points
def distance(v1, v2):
    return np.sqrt(np.sum((v2 - v1) ** 2))

# Implementing the E-step: assign points to clusters
def assign_clusters():
    for idx in range(k):
        clusters[idx]['points'] = []  # Reset points in each cluster
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)

# Function to update cluster centers
def update_cluster():
    for idx in range(k):
        pts = np.array(clusters[idx]['points'])
        if pts.shape[0] > 0:
            new_center = pts.mean(axis=0)
            clusters[idx]['center'] = new_center

# Function to plot the clusters
def plot_clusters(iteration):
    plt.figure(figsize=(8, 6))
    for idx in clusters:
        pts = np.array(clusters[idx]['points'])
        if pts.shape[0] > 0:
            plt.scatter(pts[:,0], pts[:,1], c=clusters[idx]['color'])
        plt.scatter(clusters[idx]['center'][0], clusters[idx]['center'][1], c='black', marker='X')
    plt.title(f'Iteration {iteration}')
    plt.grid(True)
    plt.show()

# Function to run the model a specified number of times and plot the progress
def run_model(iterations):
    for i in range(iterations):
        update_cluster()
        assign_clusters()
        plot_clusters(i+1)

# Run the model for a specified number of iterations
run_model(10)

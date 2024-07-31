from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dimension reduction
from sklearn.decomposition import PCA
import keras
from sklearn.discriminant_analysis import StandardScaler
from tensorflow.keras.layers import Dense, Input
import umap

# clustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, silhouette_score

features = pd.read_csv("data/sbert_embeddings.csv")
scaler = StandardScaler()
features = scaler.fit_transform(features)

labels = pd.read_csv("data/data.csv")["is_suicide"]


# Dimensionality-reduction algorithms

# PCA
# 2 principal components
# pca_2d_model = PCA(n_components=2)
# low_dim_features = pca_2d_model.fit_transform(features)

# # Visualize the PCA-transformed 2D features
# plt.figure(figsize=(8, 6))
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], alpha=0.5)
# plt.title('PCA: 2D Projection of Features')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()


# 3 principal components
# pca_3d_model = PCA(n_components=3)
# low_dim_features = pca_3d_model.fit_transform(features)

# Visualize the PCA-transformed 3D features
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# # c=labels so that point is colored based on corresponding label
# # cmap = viridis so that uses colormap that has colors distinguishable by most people, and even when in grayscale
# scatter = ax.scatter(low_dim_features[:, 0], low_dim_features[:, 1], low_dim_features[:, 2], c=labels, alpha=0.5, cmap='viridis')
# ax.set_title('PCA: 3D Projection of Features')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# cbar = fig.colorbar(scatter, ax=ax)
# cbar.set_label('Labels')
# plt.show()

# Deep Autoencoder
# encoding_dim = 2
# input_size = features.shape[1]

# input_df = Input(shape=(input_size,))
# encoded = Dense(encoding_dim, activation="relu")(input_df)
# decoded = Dense(input_size, activation="sigmoid")(encoded)

# autoencoder = keras.Model(input_df, decoded)
# encoder = keras.Model(input_df, encoded)

# autoencoder.compile(optimizer="adam", loss="mse")

# autoencoder.summary()

# # features both input and output because want to compress and then reconstruct both from original input
# autoencoder.fit(features, features,
#                 epochs=400,
#                 batch_size=64,
#                 shuffle=True,
#                 validation_data=(features, features))

# low_dim_features = encoder.predict(features)

# UMAP reducer
# need to install umap-learn and umap in pip
reducer = umap.UMAP(
    # num of neighbors used for local approximations
    n_neighbors = 45,
    min_dist = 0.7,
    n_components = 2,
    # manhattan distance is sum of absolute difference of their cooridinates
    # better than euclidean distance in high dimensional data
    metric = "manhattan"
)

low_dim_features = reducer.fit_transform(features)

# Plot the UMAP results
# plt.figure(figsize=(10, 8))
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=labels, alpha=0.5, cmap='viridis')
# plt.title('UMAP Projection of Autoencoder Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.colorbar(label='Density')
# plt.show()

# Clustering Algorithms

# Gaussian Mixture Model (GMM)
# n_components=2: GMM models the data as mixture of 2 different Gaussian distributions/clusters
# covariance_type=full: gaussian component has own full covariance matrix -> most flexibility as can have unique shape & orientation
gmm = GaussianMixture(n_components=2, covariance_type="full").fit(low_dim_features)
gmm_predictions = gmm.predict(low_dim_features)
probs = gmm.predict_proba(low_dim_features)

silhouette_gmm = silhouette_score(low_dim_features, gmm_predictions)
print(f'Silhouette Score for GMM: {silhouette_gmm}')

plt.figure(figsize=(10, 8))
plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=gmm_predictions, cmap='viridis')
plt.title('GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Create confidence score for each predicted via getting distance from centroids
centroids = gmm.means_
# Euclidean distance between two points in Euclidean space is the straight-line distance between them
distances = euclidean_distances(low_dim_features, centroids)

predicted_labels = np.empty_like(labels)
confidence_scores = np.empty(len(labels))

for cluster in np.unique(gmm_predictions):
    # Find the indices of all points in the dataset that belong to the current cluster
    cluster_indices = np.where(gmm_predictions == cluster)[0]
    
    # Get original labels of the points in the current cluster
    cluster_points_labels = labels[cluster_indices]
    
    # Get and assign most common label to the cluster
    most_common_label = Counter(cluster_points_labels).most_common(1)[0][0]
    predicted_labels[cluster_indices] = most_common_label
    
    # Calculate confidence scores based on distances
    cluster_distances = distances[cluster_indices, cluster]
    max_distance = np.max(cluster_distances)
    
    # Normalize distances to [0, 1] and invert to make smaller distances higher confidence
    confidence_scores[cluster_indices] = 1 - (cluster_distances / max_distance)

results_df = pd.DataFrame({
    'predictions': predicted_labels,
    'confidence': confidence_scores
})

# Write the DataFrame to a CSV file
results_df.to_csv('data/clustering_results.csv', index=False)


# K-means
# init=k-means++: initial centroids are distant from each other, speeds up convergence
# n_init=100: algorithm runs 100 times to find best centroid seeds
# kmeans = KMeans(n_clusters=2, init="k-means++", n_init=100).fit(low_dim_features)
# kmeans_predictions = kmeans.predict(low_dim_features)

# # KMeans scatter plots
# plt.figure(figsize=(12, 5))
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=kmeans_predictions, cmap='viridis')
# plt.title('KMeans Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# plt.show()

# # KMean's clustering silhouette score
# silhouette_kmeans = silhouette_score(low_dim_features, kmeans_predictions)
# print(f'Silhouette Score for KMeans: {silhouette_kmeans}')


# KMedoids clustering
# from sklearn_extra.cluster import KMedoids
# kmedoids = KMedoids(n_clusters=2, random_state=0)
# kmedoids.fit(features)
# labels = kmedoids.labels_

# silhouette_kmedoids = silhouette_score(features, labels)
# print(silhouette_kmedoids)


# Spectral Clustering
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse.csgraph import laplacian

# scaler = StandardScaler()
# features = scaler.fit_transform(features)

# if np.any(np.isnan(features)):
#     raise ValueError("NaN values found in the standardized data.")

# affinity_matrix = cosine_similarity(features)
# # Check for NaN values in the affinity matrix
# if np.any(np.isnan(affinity_matrix)):
#     raise ValueError("NaN values found in the affinity matrix.")

# laplacian_matrix = laplacian(affinity_matrix, normed=True)
# # Check for NaN values in the Laplacian matrix
# if np.any(np.isnan(laplacian_matrix)):
#     raise ValueError("NaN values found in the Laplacian matrix.")

# eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

# num_clusters = 2
# feature_matrix = eigenvectors[:, 1:num_clusters+1]

# kmeans = KMeans(n_clusters=num_clusters)
# clusters = kmeans.fit_predict(feature_matrix)

# # Generate new predicted labels for Spectral Clustering
# # Create confidence score for each predicted via getting distance from centroids
# centroids = kmeans.cluster_centers_
# # Euclidean distance between two points in Euclidean space is the straight-line distance between them
# distances = euclidean_distances(feature_matrix, centroids)

# predicted_labels = np.empty_like(labels)
# confidence_scores = np.empty(len(labels))

# for cluster in np.unique(clusters):
#     # Find the indices of all points in the dataset that belong to the current cluster
#     cluster_indices = np.where(clusters == cluster)[0]
    
#     # Get original labels of the points in the current cluster
#     cluster_points_labels = labels[cluster_indices]
    
#     # Get and assign most common label to the cluster
#     most_common_label = Counter(cluster_points_labels).most_common(1)[0][0]
#     predicted_labels[cluster_indices] = most_common_label
    
#     # Calculate confidence scores based on distances
#     cluster_distances = distances[cluster_indices, cluster]
#     max_distance = np.max(cluster_distances)
    
#     # Normalize distances to [0, 1] and invert to make smaller distances higher confidence
#     confidence_scores[cluster_indices] = 1 - (cluster_distances / max_distance)

# results_df = pd.DataFrame({
#     'Predicted Label': predicted_labels,
#     'Confidence Score': confidence_scores
# })

# # Write the DataFrame to a CSV file
# results_df.to_csv('data/clustering_results.csv', index=False)

# silhouette_avg = silhouette_score(feature_matrix, clusters)
# print(f'Silhouette Score: {silhouette_avg}')

# # Plotting spectral clustering results
# plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c=clusters, cmap='viridis')
# plt.xlabel('Eigenvector 1')
# plt.ylabel('Eigenvector 2')
# plt.title(f'Spectral Clustering Results\nSilhouette Score: {silhouette_avg:.2f}')
# plt.show()
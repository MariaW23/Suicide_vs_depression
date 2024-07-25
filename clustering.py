import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# dimension reduction
from sklearn.decomposition import PCA, sparse_encode
import keras
from tensorflow.keras.layers import Dense, Input
from torch import cosine_similarity
import umap

# clustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features = pd.read_csv("data/guse-embeddings.csv")
labels = pd.read_csv("data/train_data.csv")["is_suicide"]

# Dimensionality-reduction algorithms

# PCA
# 2 principal components
pca_2d_model = PCA(n_components=2)
low_dim_features = pca_2d_model.fit_transform(features)

# Visualize the PCA-transformed 2D features
plt.figure(figsize=(8, 6))
plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], alpha=0.5)
plt.title('PCA: 2D Projection of Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# 3 principal components
# pca_3d_model = PCA(n_components=4)
# low_dim_features = pca_3d_model.fit_transform(features)

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
# reducer = umap.UMAP(
#     # num of neighbors used for local approximations
#     n_neighbors = 45,
#     min_dist = 0.7,
#     n_components = 2,
#     # manhattan distance is sum of absolute difference of their cooridinates
#     # better than euclidean distance in high dimensional data
#     metric = "manhattan"
# )

# low_dim_features = reducer.fit_transform(features)

# # Plot the UMAP results
# plt.figure(figsize=(10, 8))
# plt.scatter(umap_features[:, 0], umap_features[:, 1], c=labels, alpha=0.5, cmap='viridis')
# plt.title('UMAP Projection of Autoencoder Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.colorbar(label='Density')
# plt.show()


# Clustering Algorithms

# Gaussian Mixture Model (GMM)
# n_components=2: GMM models the data as mixture of 2 different Gaussian distributions/clusters
# covariance_type=full: gaussian component has own full covariance matrix -> most flexibility as can have unique shape & orientation
# gmm = GaussianMixture(n_components=2, covariance_type="full").fit(low_dim_features)
# gmm_predictions = gmm.predict(low_dim_features)
# probs = gmm.predict_proba(low_dim_features)

# visualize GMM's proability contour
# x, y = np.meshgrid(np.linspace(min(low_dim_features[:, 0]), max(low_dim_features[:, 0]), 100),
#                    np.linspace(min(low_dim_features[:, 1]), max(low_dim_features[:, 1]), 100))
# XX = np.array([x.ravel(), y.ravel()]).T

# # Get probability densities from GMM
# probs = gmm.score_samples(XX)
# probs = probs.reshape(x.shape)

# plt.contour(x, y, probs, levels=14, linewidths=1, colors='gray')
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=gmm_predictions, cmap='viridis')
# plt.title('GMM Clustering with Contour')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

# K-means
# init=k-means++: initial centroids are distant from each other, speeds up convergence
# n_init=100: algorithm runs 100 times to find best centroid seeds
# kmeans = KMeans(n_clusters=2, init="k-means++", n_init=100).fit(low_dim_features)
# kmeans_predictions = kmeans.predict(low_dim_features)

# GMM & KMeans scatter plots
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=gmm_predictions, cmap='viridis')
# plt.title('GMM Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# plt.subplot(1, 2, 2)
# plt.scatter(low_dim_features[:, 0], low_dim_features[:, 1], c=kmeans_predictions, cmap='viridis')
# plt.title('KMeans Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# plt.show()

# compare GMM & KMean's clustering via silhouette score
# silhouette_gmm = silhouette_score(low_dim_features, gmm_predictions)
# print(f'Silhouette Score for GMM: {silhouette_gmm}')

# silhouette_kmeans = silhouette_score(low_dim_features, kmeans_predictions)
# print(f'Silhouette Score for KMeans: {silhouette_kmeans}')

# Spectral clustering
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import StandardScaler

if np.any(np.isnan(features)):
    raise ValueError("NaN values found in the standardized data.")

affinity_matrix = cosine_similarity(features)
# Check for NaN values in the affinity matrix
if np.any(np.isnan(affinity_matrix)):
    raise ValueError("NaN values found in the affinity matrix.")

laplacian_matrix = laplacian(affinity_matrix, normed=True)
# Check for NaN values in the Laplacian matrix
if np.any(np.isnan(laplacian_matrix)):
    raise ValueError("NaN values found in the Laplacian matrix.")

eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

num_clusters = 2
feature_matrix = eigenvectors[:, 1:num_clusters+1]
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(feature_matrix)

silhouette_avg = silhouette_score(feature_matrix, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Plotting the clusters (if desired)
plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.title(f'Spectral Clustering Results\nSilhouette Score: {silhouette_avg:.2f}')
plt.show()
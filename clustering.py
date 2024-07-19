import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import keras
from tensorflow.keras.layers import Dense, Input

import umap


features = pd.read_csv("data/bert-embeddings.csv")
labels = pd.read_csv("data/train_data.csv")["is_suicide"][1:]

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
pca_3d_model = PCA(n_components=3)
low_dim_features = pca_3d_model.fit_transform(features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# c=labels so that point is colored based on corresponding label
# cmap = viridis so that uses colormap that has colors distinguishable by most people, and even when in grayscale
scatter = ax.scatter(low_dim_features[:, 0], low_dim_features[:, 1], low_dim_features[:, 2], c=labels, alpha=0.5, cmap='viridis')
ax.set_title('PCA: 3D Projection of Features')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Labels')
plt.show()

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

# umap_features = reducer.fit_transform(features)

# # Plot the UMAP results
# plt.figure(figsize=(10, 8))
# plt.scatter(umap_features[:, 0], umap_features[:, 1], c=labels, alpha=0.5, cmap='viridis')
# plt.title('UMAP Projection of Autoencoder Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.colorbar(label='Density')
# plt.show()

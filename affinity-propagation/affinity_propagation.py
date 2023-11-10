# Import necessary libraries
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np

# Load a face dataset (Labeled Faces in the Wild)
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset):
# unsupervised feature extraction / dimensionality reduction
n_components = 50

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

# Projecting the input data on the eigenfaces orthonormal basis
X_pca = pca.transform(X)

# Compute Affinity Propagation
af = AffinityPropagation().fit(X_pca)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

# Plot exemplar images
plt.figure(figsize=(15, 7.5))
for i, index in enumerate(cluster_centers_indices):
    plt.subplot(2, n_clusters_//2, i + 1)
    plt.imshow(lfw_people.images[index], cmap=plt.cm.gray)
    plt.title(target_names[lfw_people.target[index]], size=12)
    plt.xticks(())
    plt.yticks(())

plt.show()


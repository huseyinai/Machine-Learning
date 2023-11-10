from sklearn.cluster import AffinityPropagation
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load a smaller subset of the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# Reduce the dimensionality of the data using PCA
n_components = 30  # Reduced number of components
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_pca = pca.fit_transform(lfw_people.data)

# Apply Affinity Propagation
af = AffinityPropagation(random_state=0)
af.fit(X_pca)
cluster_centers_indices = af.cluster_centers_indices_

# Plotting the exemplars (cluster centers)
plt.figure(figsize=(10, 5))
for i, idx in enumerate(cluster_centers_indices):
    plt.subplot(2, len(cluster_centers_indices) // 2, i + 1)
    plt.imshow(lfw_people.images[idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

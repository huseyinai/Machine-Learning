# Importing necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Iris dataset
iris = load_iris()
X = iris.data

# Optional: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Cluster centers
centers = kmeans.cluster_centers_

# Predict the cluster for each data point
y_kmeans = kmeans.predict(X_scaled)

# Evaluate the clusters
score = silhouette_score(X_scaled, y_kmeans)
print(f'Silhouette Score: {score:.2f}')

# Plotting the results
plt.figure(figsize=(14, 7))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i}')
plt.scatter(centers[:, 0], centers[:, 1], s=400, c='yellow', label='Centroids')
plt.title('Iris dataset KMeans clustering')
plt.legend()
plt.show()

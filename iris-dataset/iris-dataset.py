
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# Plot the iris dataset
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# Create a PCA object with 3 components
pca = PCA(n_components=3)

# Fit the PCA object to the iris data
pca.fit(iris.data)

# Transform the iris data into the first 3 principal components
transformed_data = pca.transform(iris.data)

# Plot the transformed data in a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=iris.target)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Print the explained variance ratio of each principal component
print(pca.explained_variance_ratio_)



# Train a decision tree classifier on the iris dataset
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# Predict the class of a new sample
new_sample = [[5.0, 3.6, 1.4, 0.2]]
predicted_class = clf.predict(new_sample)
print(predicted_class)

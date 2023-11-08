
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def iris_dataset():
    """
    This script loads the iris dataset, performs PCA on the data, and trains a decision tree classifier on the dataset.
    It also includes code to plot the iris dataset and the transformed data in a 3D scatter plot.
    """
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

    # unused but required import for doing 3d projections with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401

    # Create a PCA object with 3 components
    pca = PCA(n_components=3)

    # Fit the PCA object to the iris data
    pca.fit(iris.data)

    # Print the explained variance ratio of each principal component
    print(pca.explained_variance_ratio_)

    # Train a decision tree classifier on the iris dataset
    clf = DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)

    # Predict the class of a new sample
    new_sample = [[5.0, 3.6, 1.4, 0.2]]
    predicted_class = clf.predict(new_sample)
    print(predicted_class)


iris_dataset()
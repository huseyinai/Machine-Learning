# Iris Dataset 

## Overview
This repository contains a machine learning project that focuses on the analysis of the Iris dataset using various algorithms, including K-means clustering, K-Nearest Neighbors (KNN), Linear Regression, and Support Vector Machine (SVM).

The Iris dataset is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in 1936. It is often used for testing out machine learning algorithms and visualizations.

## Project Structure
This project is structured as follows:

- `iris_data_exploration.py`
- `decision_tree.py`
- `gradient_boosting_machines.py`
- `k_means_clustering.py`
- `k_nearest_neighbors.py`
- `linear_regression.py`
- `naive_bayes.py`
- `random_forest.py`
- `support_vector_machine.py`
- `requirements.txt`

# Iris Dataset Machine Learning Algorithms Analysis

## Overview
This project applies several machine learning algorithms to the classic Iris dataset, which contains measurements for iris flowers of three different species. The algorithms included are:

1. Decision Tree
2. Gradient Boosting Machines
3. K-Means Clustering
4. K-Nearest Neighbors
5. Linear Regression
6. Naive Bayes
7. Random Forest
8. Support Vector Machine

Each script applies its respective algorithm to the Iris dataset and seeks to classify the flowers or predict certain features accurately.

## Algorithms and Scripts

### Decision Tree (`decision_tree.py`)
A decision tree is used for classification and regression tasks. The algorithm creates a model that predicts the value of a target variable by learning simple decision rules deduced from the data features. It is intuitive and easy to interpret, making it useful in understanding the data.

- **Real-life example**: Decision trees are widely used in finance for option pricing, or in medicine to support a diagnosis based on patient's symptoms.

### Gradient Boosting Machines (`gradient_boosting_machines.py`)
Gradient Boosting Machines are a type of ensemble learning where new models are created to correct the errors made by existing models. Models are added sequentially until no further improvements can be made.

- **Real-life example**: GBMs are used in web search engines to rank pages and in ecology to model species distribution based on environmental factors.

### K-Means Clustering (`k_means_clustering.py`)
K-Means clustering is an unsupervised learning algorithm that is used to partition the dataset into K clusters. Each point belongs to the cluster with the nearest mean value.

- **Real-life example**: Market segmentation, organizing computing clusters, social network analysis, astronomical data analysis.

### K-Nearest Neighbors (`k_nearest_neighbors.py`)
K-Nearest Neighbors is a simple, instance-based learning method that stores all available cases, classifying new instances based on a similarity measure.

- **Real-life example**: Recommender systems (like Netflix or Amazon suggestions), credit scoring, and classifying genes in biological research.

### Linear Regression (`linear_regression.py`)
Linear regression is used to predict the value of a variable based on the value of another variable. The predictor variable is the variable that you are using to predict an outcome.

- **Real-life example**: In economics to predict consumption spending, healthcare for predicting prognosis, and in environmental science for estimating pollution levels.

### Naive Bayes (`naive_bayes.py`)
Naive Bayes classifiers are a family of probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

- **Real-life example**: Email filtering (spam vs. non-spam), document classification, and disease prediction.

### Random Forest (`random_forest.py`)
Random Forest is an ensemble of decision trees, typically trained with the "bagging" method. It is used for classification and regression.

- **Real-life example**: Banking (for loan risk analysis and fraud detection), e-commerce (for predicting customer behavior), and medicine (for predicting disease at an early stage).

### Support Vector Machine (`support_vector_machine.py`)
Support Vector Machine is a classification method that finds the hyperplane that maximally separates the classes in the feature space.

- **Real-life example**: Face detection, text and hypertext categorization, classification of images, bioinformatics.

Decision Tree:

Information Gain (IG): ( IG(D_p,f) = H(D_p) - \sum_{j=1}^{m} \frac{N_j}{N_p} H(D_j) )
Where ( H(D) ) is the entropy of dataset ( D ), ( D_p ) and ( D_j ) are the datasets of the parent and the j-th child node, ( N_p ) and ( N_j ) are the number of samples in the parent and j-th child node respectively.
Gradient Boosting Machines (GBM):

General Boosting Method: ( F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) )
Where ( F_m(x) ) is the boosted model at iteration ( m ), ( \gamma_m ) is the step size, and ( h_m(x) ) is the weak learner's contribution at iteration ( m ).
Models are added that correct the residuals of the existing model.
Often combined with decision trees as weak learners.
K-Means Clustering:

Objective Function: ( \displaystyle\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2 )
Where ( k ) is the number of clusters, ( S ) represents the set of all clusters, ( S_i ) is the i-th cluster, ( x ) is an observation, and ( \mu_i ) is the centroid of cluster ( S_i ).
K-Nearest Neighbors (KNN):

Classification Rule: Assign the output label based on the majority label among the ( k ) nearest neighbors.
Distance Metrics: Typically Euclidean ( d(x, y) = \sqrt{\sum (x_i - y_i)^2} ), but can be Manhattan, Minkowski, etc.
Linear Regression:

Linear Model: ( \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p )
Where ( \hat{y} ) is the predicted value, ( \beta_0 ) is the intercept, ( \beta_1, \beta_2, \ldots, \beta_p ) are the coefficients of predictor variables ( x_1, x_2, \ldots, x_p ).
Naive Bayes:

Bayes' Theorem: ( P(C_k|x) = \frac{P(C_k) P(x|C_k)}{P(x)} )
Where ( P(C_k|x) ) is the posterior probability of class ( C_k ) given features ( x ), ( P(C_k) ) is the prior probability of class ( C_k ), ( P(x|C_k) ) are the likelihoods of features ( x ) given class ( C_k ).
Random Forest:

Ensemble of Decision Trees: Combines the output of multiple decision trees through averaging (regression) or majority voting (classification).
Based on "bagging" (Bootstrap Aggregating): each tree is trained on a bootstrap sample of the data, and splitting features are randomly selected.
Support Vector Machine (SVM):

Optimization Objective: ( \displaystyle\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{n} \xi_i )
Subject to ( y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i ) and ( \xi_i \geq 0)
Where ( \mathbf{w} ) defines the margin width, ( b ) is the bias term, ( C ) is the penalty term for misclassification, ( \xi_i ) are the slack variables allowing soft-margin, ( y_i ) and ( \mathbf{x}_i ) are the labels and features of instances.

## Installation and Usage
To run the notebooks, please follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt` by running:
pip install -r requirements.txt
in your terminal or command prompt.
3. Open the desired notebook using Jupyter Notebook or JupyterLab.
4. Run the cells in the notebook to see the algorithms in action.

## Data
The Iris dataset used in this project can be loaded through the `sklearn.datasets` module, or it can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris) for download.

## Contributing
Contributions to the project are welcome! Please feel free to fork the repository, make changes, and create a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
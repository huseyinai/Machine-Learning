from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import json

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier()

# Fit the model
tree_classifier.fit(X_train, y_train)

# Convert tree to a dictionary
def tree_to_json(decision_tree, feature_names):
    tree_ = decision_tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left = recurse(tree_.children_left[node])
            right = recurse(tree_.children_right[node])
            return {"name": name, "threshold": threshold, "left": left, "right": right}
        else:
            return {"value": tree_.value[node].tolist()}

    return recurse(0)

# Use the function to convert the tree to JSON
tree_in_json = tree_to_json(tree_classifier, iris.feature_names)

# Save to a JSON file
with open('tree.json', 'w') as f:
    json.dump(tree_in_json, f)

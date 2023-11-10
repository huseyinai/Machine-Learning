import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset
# Replace 'your_data.csv' with the path to your dataset
# The dataset should have two columns: 'text' for the email content and 'label' for spam/ham
df = pd.read_csv('C:\Github\Machine-Learning\email-spam\data.csv')

# Preprocess the data
# Convert labels to a binary format
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

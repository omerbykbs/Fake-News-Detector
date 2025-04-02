import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data_preprocessing import load_and_preprocess_data

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load and preprocess data
data = load_and_preprocess_data()

# Split data
X = data['text']
y = data['class']

# Vectorization
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Save vectorizer
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, random_state=42)

# Initialize models
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(random_state=0)
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name} Report:\n", classification_report(y_test, predictions))
    joblib.dump(model, f"models/{name}.pkl")

print("\nAll models trained and saved successfully.")

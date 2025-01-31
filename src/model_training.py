import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

import joblib  # For saving/loading models

def train_models(df: pd.DataFrame):
    """
    Given a DataFrame with columns "text" (preprocessed) and "class",
    trains several models and returns them in a dictionary.
    """
    X = df["text"]
    y = df["class"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test = vectorizer.transform(X_test)
    
    # We can store models in a dict
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(random_state=0),
        "RandomForest": RandomForestClassifier(random_state=0)
    }
    
    trained_models = {}
    for model_name, model in models.items():
        model.fit(Xv_train, y_train)
        preds = model.predict(Xv_test)
        print(f"=== {model_name} ===")
        print(f"Score: {model.score(Xv_test, y_test):.4f}")
        print(classification_report(y_test, preds))
        
        # Save trained model and the vectorizer
        # In a real scenario, you might choose only the best model.
        trained_models[model_name] = model
    
    return vectorizer, trained_models

def save_model(model, vectorizer, model_name: str = "models/model"):
    """
    Saves the trained model and vectorizer to disk using joblib.
    """
    joblib.dump(model, f"{model_name}.pkl")
    joblib.dump(vectorizer, f"{model_name}_vectorizer.pkl")

def load_model(model_name: str = "model"):
    """
    Loads the trained model and vectorizer from disk.
    """
    model = joblib.load(f"{model_name}.pkl")
    vectorizer = joblib.load(f"{model_name}_vectorizer.pkl")
    return model, vectorizer

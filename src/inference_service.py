import joblib
import os
from src.data_preprocessing import word_opt
from sklearn.metrics import accuracy_score
from src.data_preprocessing import load_and_preprocess_data
import time

_cached_data = None

# Load default model and vectorizer
def load_model_and_vectorizer(model_name="LogisticRegression"):
    model_path = f"models/{model_name}.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize with default model
model, vectorizer = load_model_and_vectorizer()

def model_supports_proba():
    return hasattr(model, "predict_proba")

def predict_fake_news(title, text):
    combined_text = f"{title} {text}"
    cleaned_text = word_opt(combined_text)
    vect_text = vectorizer.transform([cleaned_text])
    '''
    prediction = model.predict(vect_text)[0]
    return "FAKE News" if prediction == 0 else "NOT Fake News"'''
    
    if model_supports_proba():
        prob = model.predict_proba(vect_text)[0]
        return {
            "prediction": "FAKE News" if prob[0] > 0.5 else "NOT Fake News",
            "confidence": round(max(prob), 4)
        }
    else:
        prediction = model.predict(vect_text)[0]
        return {
            "prediction": "FAKE News" if prediction == 0 else "NOT Fake News",
            "confidence": "N/A (predict_proba not supported)"
        }

def set_model(model_name):
    global model, vectorizer
    try:
        model, vectorizer = load_model_and_vectorizer(model_name)
        return {
            "status": f"Model switched to {model_name}",
            "supports_proba": model_supports_proba()
        }
    except Exception as e:
        return {"error": str(e)}

def evaluate_model(model, vectorizer):
    global _cached_data
    if _cached_data is None:
        print("Loading full dataset...")
        _cached_data = load_and_preprocess_data()
    data = _cached_data
    
    start = time.time()
    X = vectorizer.transform(data['text'])
    y = data['class']
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Evaluated model in {time.time() - start:.2f}s")
    return acc

def get_all_model_accuracies():
    model_names = ["LogisticRegression", "DecisionTree", "GradientBoosting", "RandomForest"]
    accuracies = {}

    for name in model_names:
        try:
            mdl, vec = load_model_and_vectorizer(name)
            acc = evaluate_model(mdl, vec)
            accuracies[name] = round(acc, 4)
        except Exception as e:
            accuracies[name] = f"Error: {str(e)}"

    return accuracies

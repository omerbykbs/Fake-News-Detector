import joblib
import os
from src.data_preprocessing import word_opt

# Load default model and vectorizer
def load_model_and_vectorizer(model_name="LogisticRegression"):
    model_path = f"models/{model_name}.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize with default model
model, vectorizer = load_model_and_vectorizer()

# Predict function
def predict_fake_news(title, text):
    combined_text = f"{title} {text}"
    cleaned_text = word_opt(combined_text)
    vect_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vect_text)[0]
    return "FAKE News" if prediction == 0 else "NOT Fake News"

# Switch model dynamically
def set_model(model_name):
    global model, vectorizer
    try:
        model, vectorizer = load_model_and_vectorizer(model_name)
        return {"status": f"Model switched to {model_name}"}
    except Exception as e:
        return {"error": str(e)}

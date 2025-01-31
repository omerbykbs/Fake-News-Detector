from flask import Flask, request, jsonify
from src.data_preprocessing import clean_text
import joblib

app = Flask(__name__)

def load_model_and_vectorizer(model_name):
    model_path = f"models/{model_name}.pkl"
    vectorizer_path = f"models/{model_name}_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Default model to load
default_model_name = os.getenv("DEFAULT_MODEL", "LogisticRegression")
model, vectorizer = load_model_and_vectorizer(default_model_name)

def output_label(n):
    return "FAKE News" if n == 0 else "NOT Fake News"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Expect JSON like: { "text": "Some news headline or content here" }
    user_text = data.get("text", "")
    
    cleaned = clean_text(user_text)
    vect_input = vectorizer.transform([cleaned])
    pred = model.predict(vect_input)[0]
    
    return jsonify({
        "prediction": output_label(pred),
        "label": int(pred)
    })
    
@app.route('/set_model', methods=['POST'])
def set_model():
    global model, vectorizer
    data = request.get_json(force=True)
    model_name = data.get("model_name", default_model_name)
    try:
        model, vectorizer = load_model_and_vectorizer(model_name)
        return jsonify({"status": f"Model switched to {model_name}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
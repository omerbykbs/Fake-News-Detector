from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference_service import predict_fake_news, set_model
from typing import List
from src.inference_service import get_all_model_accuracies

app = FastAPI()

class NewsArticle(BaseModel):
    title: str
    text: str

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}

@app.post("/predict/")
def predict_news(article: NewsArticle):
    try:
        prediction = predict_fake_news(article.title, article.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batch_predict/")
def batch_predict(articles: List[NewsArticle]):
    try:
        return {"predictions": [predict_fake_news(article.title, article.text) for article in articles]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracies/")
def model_accuracies():
    try:
        print("Starting full model evaluation...")
        return get_all_model_accuracies()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model/")
def change_model(model_name: str):
    response = set_model(model_name)
    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

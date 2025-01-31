import os
import pandas as pd
from src.data_preprocessing import load_and_merge_data, preprocess_dataframe
from src.model_training import train_models, save_model

def main():
    # 1. Load & Merge
    df = load_and_merge_data(data_dir="data")
    
    df = preprocess_dataframe(df, text_col="text")
    
    vectorizer, trained_models = train_models(df)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model in trained_models.items():
        save_model(model, vectorizer, model_name=f"{models_dir}/{model_name}")

if __name__ == "__main__":
    main()

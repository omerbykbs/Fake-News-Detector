import os
import re
import string
import pandas as pd

def load_and_merge_data(data_dir: str = "data"):
    """
    Loads Fake.csv and True.csv from data_dir, merges into a single DataFrame.
    Returns the combined DataFrame with class labels and minimal cleaning.
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    fake_news = pd.read_csv(fake_path)
    true_news = pd.read_csv(true_path)

    fake_news["class"] = 0
    true_news["class"] = 1

    # Drop last 10 rows to create test split manually if you wish
    # or we can rely entirely on train_test_split.
    # (Your original code manually removed rows - that’s optional.)
    
    data_merge = pd.concat([fake_news, true_news], axis=0)
    data_merge = data_merge.sample(frac=1).reset_index(drop=True)
    
    # Drop columns you don’t need
    if set(["title","subject","date"]).issubset(data_merge.columns):
        data_merge.drop(columns=["title", "subject", "date"], inplace=True)

    return data_merge

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Applies text cleaning to the specified column.
    """
    df[text_col] = df[text_col].apply(clean_text)
    return df
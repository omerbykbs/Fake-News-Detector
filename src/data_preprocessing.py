import pandas as pd
import re
import string
import os

def word_opt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def load_and_preprocess_data():
    base_dir = os.getcwd()
    fake_path = os.path.join(base_dir, 'data', 'Fake.csv')
    true_path = os.path.join(base_dir, 'data', 'True.csv')

    # Load datasets
    fake_news = pd.read_csv(fake_path)
    true_news = pd.read_csv(true_path)

    # Assign classes
    fake_news["class"] = 0
    true_news["class"] = 1

    # Merge datasets
    data_merge = pd.concat([fake_news, true_news], axis=0)
    data_merge = data_merge.drop(columns=['title', 'subject', 'date'], axis=1)
    data_merge = data_merge.sample(frac=1).reset_index(drop=True)

    # Apply text cleaning
    data_merge['text'] = data_merge['text'].apply(word_opt)

    return data_merge

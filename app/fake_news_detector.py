import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string 
import os

base_dir = os.getcwd()
relative_path_fake = os.path.join(base_dir, 'data', 'Fake.csv')
relative_path_true = os.path.join(base_dir, 'data', 'True.csv')

# Fake Data
fake_news = pd.read_csv(relative_path_fake)

# True Data
true_news = pd.read_csv(relative_path_true)

fake_news.head()

true_news.head()

fake_news["class"] = 0
true_news["class"] = 1

fake_news.shape, true_news.shape

test_fake_news = fake_news.tail(10)
for i in range(23480, 23470, -1):
    fake_news.drop([i], axis= 0, inplace=True)
    
test_true_news = true_news.tail(10)
for i in range(21416, 21406, -1):
    true_news.drop([i], axis=0, inplace=True)

fake_news.shape, true_news.shape

test_fake_news["class"] = 0
test_true_news["class"] = 1

test_fake_news.head(10)

test_true_news.head(10)

data_merge = pd.concat([fake_news, true_news], axis=0)
data_merge.head(10)

data_merge.columns

data = data_merge.drop(columns=['title','subject', 'date'], axis=1)

data.isnull().sum()

print(data['class'].value_counts())

data = data.sample(frac=1)

data.head(10)

data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

data.columns

data.head

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

data['text'] = data['text'].apply(word_opt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

print("LR Report\n",classification_report(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print("DT Report\n",classification_report(y_test, pred_dt))

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)

GB.score(xv_test, y_test)

print("GB Report\n",classification_report(y_test, pred_gb))

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)


pred_rf = RF.predict(xv_test)

RF.score(xv_test, y_test)

print("RF Report\n",classification_report(y_test, pred_rf))

def output_label(n):
    if n == 0:
        return "FAKE News"
    elif n == 1:
        return "NOT A Fake News"
    
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_opt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print(f"\n\nLR Prediction: {output_label(pred_LR[0])} \nDT Prediction: {output_label(pred_DT[0])} \nGB Prediction: {output_label(pred_GB[0])} \nRF Prediction: {output_label(pred_RF[0])}")

news = str(input())
manual_testing(news)




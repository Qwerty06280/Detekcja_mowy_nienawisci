## Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# regex - cleaning
import re
# lematyzacja
import spacy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# SVM
from sklearn.svm import SVC
# Naive-Bayes
from sklearn.naive_bayes import MultinomialNB

##### Stop words - Polish
with open(r'C:\Users\Chill\Desktop\INZYNIERKA\dane\polish_stopwords.txt', 'r', encoding='utf-8') as file:
    polish_stop_words = [row.strip() for row in file]
nlp = spacy.load('pl_core_news_sm')  # more precise - pl_core_news_lg / less precise & quick pl_core_news_sm

def read_sample_data(dataset: str = 'dataset_zwroty'):
    # read data
    file_path_conc = r'C:\Users\Chill\Desktop\INZYNIERKA\dane\found_internet\CONCATENATED_DATA.xlsx'
    df = pd.read_excel(file_path_conc)
    #'dataset_poleval', 'dataset_zwroty', 'dataset_wykop'
    if dataset is None:
        pass
    else:
        df = df[df['source'] == dataset]
    return df

def lemmatize_text(text):
    # Lematyzacja
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def prepare_data(data):
    # Cleaning
    # replace with empty string (delete) any character that is not: whitespace, a number, a character a-z or underscore
    df = data.copy()
    try:
        df['Clean_comment'] = df['Comment'].str.lower().apply(lambda row: re.sub(r'[^\w\s]', '', row))
        df['Final_comment'] = df['Clean_comment'].apply(lemmatize_text)
    except:
        raise Exception("Input 'df' missing column 'Comment'")
    return df

def Vectorize(method='Bag of Words', stop_words=None):
    # Tokenization & Vectorization
    if method == 'Bag of Words':
        vectorizer = CountVectorizer(lowercase=True, stop_words=polish_stop_words)  # TODO parametry
    elif method == 'TF-IDF':
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=polish_stop_words)  # TODO parametry
    else:
        raise ValueError("Method not found")
    return vectorizer

def make_predictions(data, comments_col='Final_comment', target_col='Is_toxic', vectoraizer_name='Bag of Words',
                     model_name='Logistic Regression', stop_words=None, test_size=0.2, n_splits=5):
    vectorizer = Vectorize(method=vectoraizer_name, stop_words=stop_words)
    X = vectorizer.fit_transform(data[comments_col])
    # Split data to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, data[target_col], test_size=test_size, random_state=42)

    # MODEL
    # Logistic regression
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    # SVM
    elif model_name == 'SVM':
        model = SVC()
    # Naive-Bayes
    elif model_name == 'Naive-Bayes':
        model = MultinomialNB()
    else:
        raise ValueError('Wrong model_name')

    # cross fold validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=n_splits)

    # TRAIN MODEL
    model.fit(X_train, y_train)
    # PREDICT
    predictions = model.predict(X_test)

    return y_test, predictions, cv_scores

def visualize_results(y_test, predictions, cv_scores, model_name: str, vectorizer_name: str):
    cm = confusion_matrix(y_test, predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    print(f"MODEL - {model_name.upper()}")
    print(f"VECTORIZER - {vectorizer_name.upper()}")
    print("Dokładność: {}%".format(np.round(accuracy_score(y_test, predictions) * 100, 2)))
    print(f"Średnia dokładność z walidacji krzyżowej: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    print("Raport klasyfikacji:\n", classification_report(y_test, predictions))
    cm_display.plot(cmap=plt.cm.Greens)
    plt.show()
    print("\n--------------------------------------------------------\n")

data = read_sample_data('dataset_zwroty')
df = prepare_data(data)

# BoW & LogReg
logReg_pred = make_predictions(data=df,
                               comments_col='Final_comment',
                               target_col='Is_toxic',
                               vectoraizer_name='Bag of Words',
                               model_name='Logistic Regression',
                               stop_words=polish_stop_words,
                               test_size=0.2,
                               n_splits=5)

visualize_results(y_test=logReg_pred[0],
                  predictions=logReg_pred[1],
                  cv_scores=logReg_pred[2],
                  model_name='Logisitic Regression',
                  vectorizer_name='Bag of Words')


# df_added_pred = df.loc[list(logReg_pred[0].index)]
# df_added_pred['prediction'] = logReg_pred[1]
# df_added_pred[df_added_pred['Is_toxic'] != df_added_pred['prediction']]


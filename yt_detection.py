# Import libraries
import numpy as np
import pandas as pd
# regex - cleaning
import re
# lemmatization
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
from typing import Tuple
from joblib import load


def load_components(lemmatization_method: str = 'quick') -> Tuple:
    """
    Loads neccesary components: polish_stop_words from a .txt file and data for lemmatization
    :param lemmatization_method: pick which data should be loaded, smaller set but much quicker to process, or more advanced but slower
    :return: polish_stop_words, nlp
    """
    with open(r'C:\Users\Chill\Desktop\INZYNIERKA\dane\polish_stopwords.txt', 'r', encoding='utf-8') as file:
        polish_stop_words = [row.strip() for row in file]
    if lemmatization_method == 'quick':
        nlp = spacy.load('pl_core_news_sm')  # more precise - pl_core_news_lg / less precise & quick pl_core_news_sm
    elif lemmatization_method == 'precise':
        nlp = spacy.load('pl_core_news_lg')  # more precise - pl_core_news_lg / less precise & quick pl_core_news_sm
    else:
        raise ValueError("Wrong argument value")
    return polish_stop_words, nlp

def read_sample_data(dataset: str = 'dataset_zwroty') -> pd.DataFrame:
    """
    Reads sample data for the purpose of development, experimenting, modelling
    :param dataset: sample data consists of 3 datasets. if dataset = None, then all data gets loaded. Other options:
    ['dataset_poleval', 'dataset_zwroty', 'dataset_wykop']
    :return: dataframe with loaded data
    """
    # read data
    file_path_conc = r'C:\Users\Chill\Desktop\INZYNIERKA\dane\found_internet\CONCATENATED_DATA.xlsx'
    df = pd.read_excel(file_path_conc)

    if dataset is None:
        pass
    else:
        df = df[df['source'] == dataset]
    return df

def lemmatize_text(text,nlp):
    """
    Performs lemmatization of given text
    :param text: data that needs to be lemmatized
    :return: lemmatized text
    """
    # Lematyzacja
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def prepare_data(data,nlp) -> pd.DataFrame:
    """
    Performs preprocessing - adds two new columns to dataset. Cleans strings & applies lemmatization
    :param data: dataframe with column 'Comment'
    :return: dataframe with two new columns
    """
    # Cleaning
    # replace with empty string (delete) any character that is not: whitespace, a number, a character a-z or underscore
    df = data.copy()
    try:
        df['Clean_comment'] = df['Comment'].str.lower().apply(lambda row: re.sub(r'[^\w\s]', '', row))
        df['Final_comment'] = df['Clean_comment'].apply(lambda x: lemmatize_text(x, nlp))
    except:
        raise Exception("Input 'df' missing column 'Comment'")
    return df

def Vectorize(method :str='Bag of Words', stop_words=None):
    """
    :param method: Bag of Words or TF-IDF
    :param stop_words: pass polish_stop_words or None
    :return: Vectroizer
    """
    # Tokenization & Vectorization
    if method == 'Bag of Words':
        vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)  # TODO parametry
    elif method == 'TF-IDF':
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words)  # TODO parametry
    else:
        raise ValueError("Method not found")
    return vectorizer

def load_model_and_vectorizer(vectoraizer_name='Bag of Words', model_name='Logistic Regression'):
    # VECTORAIZER
    if vectoraizer_name == 'Bag of Words':
        vect = 'BoW'
    elif vectoraizer_name == 'TF-IDF':
        vect = 'TFIDF'
    else:
        raise ValueError('Vectorizer name is invalid')
    # MODEL
    if model_name == 'Logistic Regression':
        mod = 'LogReg'
    elif model_name == 'SVM':
        mod = 'SVM'
    elif model_name == 'Naive-Bayes':
        mod = 'NB'
    else:
        raise ValueError("Model name is invalid")
    mod_vec_str = f"models_trained/model_{mod}_{vect}"
    loaded_model = load(f'{mod_vec_str}.joblib')
    loaded_vectorizer = load(f'models_trained/{vect}.joblib')
    return loaded_model, loaded_vectorizer

def make_predictions(model, comments, vectoraizer, comments_col='Final_comment', stop_words=None, test_size=0.2, n_splits=5):
    """
    """
    # vectorization
    X = vectoraizer.transform(comments[comments_col])
    # prediciton
    predictions = model.predict(X)
    return predictions

def youtube_detection(yt_comment, vect_name= 'Bag of Words', model_name= 'Logistic Regression', lemmatization= 'quick'):
    # load model and vectorizer
    model, vectorizer = load_model_and_vectorizer(vect_name, model_name)
    # load components
    polish_stop_words, nlp = load_components(lemmatization_method=lemmatization)
    # prepare data - processing
    yt_df = prepare_data(yt_comment,nlp)
    yt_predicted = make_predictions(model=model,
                                    comments=yt_df,
                                    vectoraizer=vectorizer,
                                    comments_col='Final_comment',
                                    stop_words=polish_stop_words)
    return yt_predicted
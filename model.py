# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from joblib import dump


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

def lemmatize_text(text):
    """
    Performs lemmatization of given text
    :param text: data that needs to be lemmatized
    :return: lemmatized text
    """
    # Lematyzacja
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def prepare_data(data) -> pd.DataFrame:
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
        df['Final_comment'] = df['Clean_comment'].apply(lemmatize_text)
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

def make_predictions(data, comments_col='Final_comment', target_col='Is_toxic', vectoraizer_name='Bag of Words',
                     model_name='Logistic Regression', stop_words=None, test_size=0.2, n_splits=5):
    """
    Core function, performs vectorization with function Vectorize(), splits data into training and testing subsets,
    creates model, perfors cross fold validation, trains model and makes predictions for test data
    :param data: data
    :param comments_col: column that we are going to use
    :param target_col: column with labels
    :param vectoraizer_name: pick vectorization method - 'Bag of Words'or 'TF-IDF'
    :param model_name: pick model- 'Logistic Regression', 'SVM' or 'Naive-Bayes'
    :param stop_words: polish_stop_words or None
    :param test_size: default =0.2
    :param n_splits: n_splits for cross fold validation
    :return: y_test, predictions, cv_scores, model
    """
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

    return y_test, predictions, cv_scores, model

def visualize_results(y_test, predictions, cv_scores, model_name: str, vectorizer_name: str):
    """
    Creates and displays confusion matrix for given data
    :param y_test: true labels
    :param predictions: predicted labels
    :param cv_scores: cross fold validation scored
    :param model_name: name of model used
    :param vectorizer_name: name of vectorization method used
    """
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

# polish_stop_words, nlp = load_components(lemmatization_method='precise')
# data = read_sample_data(None)
# df = prepare_data(data)
#
# # BoW & LogReg
# logReg_pred = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'Bag of Words',
#                             model_name = 'Logistic Regression',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)
#
# svm_pred = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'Bag of Words',
#                             model_name = 'SVM',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)
#
# nb_pred = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'Bag of Words',
#                             model_name = 'Naive-Bayes',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)
#
# # TF-IDF
# logReg_pred_idf = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'TF-IDF',
#                             model_name = 'Logistic Regression',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)
#
# svm_pred_idf = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'TF-IDF',
#                             model_name = 'SVM',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)
#
# nb_pred_idf = make_predictions(data = df,
#                             comments_col = 'Final_comment',
#                             target_col  = 'Is_toxic',
#                             vectoraizer_name = 'TF-IDF',
#                             model_name = 'Naive-Bayes',
#                             stop_words = polish_stop_words,
#                             test_size = 0.2,
#                             n_splits=5)


visualize_results(y_test=logReg_pred[0],
                  predictions=logReg_pred[1],
                  cv_scores=logReg_pred[2],
                  model_name='Logisitic Regression',
                  vectorizer_name='Bag of Words')


# # TRAINING MODELS
# model_LogReg_BoW = logReg_pred[3]
# dump(model_LogReg_BoW, 'models_trained/model_LogReg_BoW.joblib')
#
# model_SVM_BoW = svm_pred[3]
# dump(model_SVM_BoW, 'models_trained/model_SVM_BoW.joblib')
#
# model_NB_BoW = nb_pred[3]
# dump(model_NB_BoW, 'models_trained/model_NB_BoW.joblib')
#
# model_LogReg_TFIDF = logReg_pred_idf[3]
# dump(model_LogReg_TFIDF, 'models_trained/model_LogReg_TFIDF.joblib')
#
# model_SVM_TFIDF = svm_pred_idf[3]
# dump(model_SVM_TFIDF, 'models_trained/model_SVM_TFIDF.joblib')
#
# model_NB_TFIDF = nb_pred_idf[3]
# dump(model_NB_TFIDF, 'models_trained/model_NB_TFIDF.joblib')
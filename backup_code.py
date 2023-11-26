# polish_stop_words, nlp = load_components(lemmatization_method='precise')
# data = read_sample_data(None)
# df = prepare_data(data)

## FITTING VECTORIZERS
# bow = Vectorize('Bag of Words', stop_words=polish_stop_words)
# X = bow.fit_transform(df["Final_comment"])
# tfidf = Vectorize('TF-IDF', stop_words=polish_stop_words)
# X2 = tfidf.fit_transform(df["Final_comment"])
# dump(bow, 'models_trained/bow.joblib')
# dump(tfidf, 'models_trained/bow.joblib')

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


# visualize_results(y_test=logReg_pred[0],
#                   predictions=logReg_pred[1],
#                   cv_scores=logReg_pred[2],
#                   model_name='Logisitic Regression',
#                   vectorizer_name='Bag of Words')


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


# # PARAMETR VECTORIZER
# label_vect_name = tk.Label(window, text="Wybierz metodę wektoryzacji:")
# label_vect_name.pack(padx=10, pady=(10,0))
# tk.Radiobutton(window, text="Bag of Words", variable=vect_name_var, value="Bag of Words").pack()
# tk.Radiobutton(window, text="TF-IDF", variable=vect_name_var, value="TF-IDF").pack()
#
# # PARAMETR MODEL
# label_model_name = tk.Label(window, text="Wybierz model uczenia maszynowego:")
# label_model_name.pack(padx=10, pady=(10,0))
# tk.Radiobutton(window, text="Logistic Regression", variable=model_name_var, value="Logistic Regression").pack()
# tk.Radiobutton(window, text="SVM", variable=model_name_var, value="SVM").pack()
# tk.Radiobutton(window, text="Naive Bayes", variable=model_name_var, value="Naive Bayes").pack()
#
# # PARAMETR LEMMATIZATION
# label_lemmatization = tk.Label(window, text="Wybierz precyzję lematyzacji:")
# label_lemmatization.pack(padx=10, pady=(10,0))
# tk.Radiobutton(window, text="Precyzyjna", variable=lemmatization_var, value="precise").pack()
# tk.Radiobutton(window, text="Szybka", variable=lemmatization_var, value="quick").pack()

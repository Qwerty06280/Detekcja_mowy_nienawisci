import GUI

def main():
    pass

if __name__ == "__main__":
    main()

#pyinstaller --onefile --add-data "models_trained/BoW.joblib;models_trained"
# --add-data "models_trained/model_LogReg_BoW.joblib;models_trained"
# --add-data "models_trained/model_LogReg_TFIDF.joblib;models_trained"
# --add-data "models_trained/model_NB_BoW.joblib;models_trained"
# --add-data "models_trained/model_NB_TFIDF.joblib;models_trained"
# --add-data "models_trained/model_SVM_BoW.joblib;models_trained"
# --add-data "models_trained/model_SVM_TFIDF.joblib;models_trained"
# --add-data "models_trained/TFIDF.joblib;models_trained"
# --add-data "C:/ProgramData/Anaconda3/envs/pythonProject/lib/site-packages/pl_core_news_lg;spacy/data_lemmatization"
# --add-data "C:/ProgramData/Anaconda3/envs/pythonProject/lib/site-packages/pl_core_news_sm;spacy/data_lemmatization" main.py
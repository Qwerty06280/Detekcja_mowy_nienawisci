# test_my_functions.py
import unittest
from YT_API import yt_extract
from yt_detection import load_model_and_vectorizer

class TestKeyFunctions(unittest.TestCase):
    def test_api_connection(self):
        """
        test check if the connection with Youtube API can be estabilished
        """
        example_link = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

        # Sprawdzenie, czy funkcja wykonuje się bez rzucania wyjątku
        try:
            yt_extract(example_link)
        except Exception:
            self.fail("YouTube connection failed")

    def test_model_vectorizer(self):
        """
        test checks if models are able to be loaded
        """
        try:
            load_model_and_vectorizer(vectoraizer_name='Bag of Words', model_name='Logistic Regression')
            load_model_and_vectorizer(vectoraizer_name='TF-IDF', model_name='SVM')
            load_model_and_vectorizer(vectoraizer_name='Bag of Words', model_name='Naive-Bayes')
        except Exception:
            self.fail("Models not found")

if __name__ == '__main__':
    unittest.main()
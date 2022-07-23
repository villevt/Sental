import nltk
import numpy as np
import string


def clean_text(sentence):
    """ 
    Cleans a list of sentences:
        - Normalizes text to lowercase
        - Removes stopwords and punctuation
        - Tokenizes words
    """

    try:
        stopwords = nltk.corpus.stopwords.words("english")
        nltk.tokenize.word_tokenize("test test")
    except:
        nltk.download("stopwords")
        nltk.download("punkt")
        stopwords = nltk.corpus.stopwords.words("english")

    return np.array([w.lower() for w in nltk.tokenize.word_tokenize(sentence) 
        if w not in stopwords and w not in string.punctuation])

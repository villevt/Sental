import nltk
import numpy as np
import string
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import ComplementNB

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

class NLPClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, feature_extractor=CountVectorizer(), estimator=ComplementNB(), ngram_range=(1, 1), min_df=0):
        """ 
        Inits the model.
        Arguments:
            feature_extractor: Method to use for transforming feature extraction. Supports some sklearn extractors:
                - CountVectorizer()
                - TfidfVectorizer
            estimator: sklearn estimator to use. Options:
                - ComplementNB()
                - GaussianNB()
                - MultinomialNB()
                - LinearSVC()
                - SGDClassifier()
            ngram_range: Range of ngrams to produce as features
        """

        self.feature_extractor = feature_extractor
        self.ngram_range = ngram_range
        self.min_df = min_df

        self.estimator = estimator

    def fit(self, X, y):
        """
        Fits the the model.
        Arguments:
            X:  Input data
            y:  Input labels
        """
        self.feature_extractor.set_params(tokenizer=clean_text, ngram_range=self.ngram_range, min_df=self.min_df)

        X_processed = self.feature_extractor.fit_transform(X.copy()).toarray()
        self.estimator = self.estimator.fit(X_processed, y)

    def score(self, X, y):
        """
        Evaluates the accuracy of the data.
        Arguments:
            - X: Real labels
            - y: Real labels

        Returns:
            Balanced accuracy score of predicted labels
        """

        return balanced_accuracy_score(y, self.predict(X))

    def predict(self, X):
        """
        Generates predictions based on data.
        Arguments:
            X: (unseen) data
        """

        X = self.feature_extractor.transform(X.copy()).toarray()
        y = self.estimator.predict(X)
        return y
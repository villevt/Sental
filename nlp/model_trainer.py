import argparse
import joblib
from math import ceil, floor
import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import textutils

class Model(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, feature_extractor=CountVectorizer(), estimator=ComplementNB(), ngram_range=(1, 1)):
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

        self.estimator = estimator

    def fit(self, X, y):
        """
        Fits the the model.
        Arguments:
            X:  Input data
            y:  Input labels
        """
        self.feature_extractor.set_params(tokenizer=textutils.clean_text, ngram_range=self.ngram_range)

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


def load_data(datasets, train_size=0.8):
    """
    Loads a dataset for training the model.
    Arguments:
        datasets:   Datasets to use in training. Takes in tab-delimited text files with sentences on first column and binary sentiment on the second.
        partition:  Should the dataset be partitioned, i.e. should the entire data be loaded into memory at once?
        train_size: Size of training data set as a fraction of the dataset
    """
    # Load data
    data = np.concatenate(
        [np.loadtxt(d, dtype=str, delimiter="\t") for d in datasets])

    # Shuffle data
    np.random.default_rng().shuffle(data)

    # Split into training and testing data
    trainIdx = floor(data.shape[0] * train_size)
    train = data[:trainIdx, :]
    test = data[trainIdx:, :]

    # Set X and y
    X_train, y_train = train[:, 0], train[:, 1].astype(int)
    X_test, y_test = test[:, 0], test[:, 1].astype(int)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", nargs="+",
                        help="list of .txt files to use in training")
    parser.add_argument("-d", nargs="?", default="/data/",
                        help="directory for training data files")
    parser.add_argument("-s", nargs="?", default=0.8,
                        help="train set size")

    args = parser.parse_args()

    # Load datasets from the given folder and files
    datasets = [os.getcwd() + args.d + f for f in args.f]

    # Initialize model
    model = Model()
    X_train, X_test, y_train, y_test = load_data(
        datasets, train_size=float(args.s))


    gs = GridSearchCV(Model(), param_grid={
        "feature_extractor": [CountVectorizer(), TfidfVectorizer()],
        "estimator": [ComplementNB(), GaussianNB(), MultinomialNB(), LinearSVC(class_weight="balanced"), SGDClassifier(class_weight="balanced")],
        "ngram_range": [(1, 1), (1, 2), (1, 3)]}, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)

    print(gs.cv_results_)

    print(gs.best_estimator_)

    print(f"Selected parameters: {gs.best_params_}")
    print(f"Mean score on validation set: {gs.best_score_}")
    print(f"Balanced accuracy score on test set of {ceil((1 - float(args.s)) * 100)}%: {gs.score(X_test, y_test)}")

    joblib.dump(gs.best_estimator_,  "sentiment_classifier.joblib.pkl")
import argparse
import joblib
from math import ceil, floor
from nlpclassifier import NLPClassifier
import numpy as np
import os
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC


def load_data(datasets, train_size=0.8):
    """
    Loads a dataset for training the model.
    Arguments:
        datasets:   Datasets to use in training. Takes in tab-delimited text files with sentences on first column and binary sentiment on the second.
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


def load_lexicon(file):
    """
    Loads a lexicon
    Arguments:
        file:  File to take lexicon from. Takes in tab-delimited text files with sentences on first column and sentiment on the second.
    """
    # Load data
    data = np.loadtxt(file, dtype=str, delimiter="\t")

    # Turn into dict
    return {key: int(val) for key, val in data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", nargs="+",
                        help="list of .txt files to use in training")
    parser.add_argument("-d", nargs="?", default="/data/",
                        help="directory for training data files")
    parser.add_argument("-s", nargs="?", default=0.8,
                        help="train set size")
    parser.add_argument("-l", nargs="?",
                        help="lexicon .txt file to use in training")

    args = parser.parse_args()

    # Load datasets from the given folder and files
    datasets = [os.getcwd() + args.d + f for f in args.f]
    X_train, X_test, y_train, y_test = load_data(
        datasets, train_size=float(args.s))

    if args.l:
        lexicon = load_lexicon(os.getcwd() + args.d + args.l)
    else:
        lexicon = None

    # Use a select set of estimators
    estimators_minmax = [ComplementNB(), GaussianNB(), MultinomialNB(), LinearSVC(class_weight="balanced"), 
        SGDClassifier(class_weight="balanced")]

    estimators_scaled = [LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, tol=1e-2, C=0.25)]

    grid_minmax = {
        "estimator": [CalibratedClassifierCV(e, method="sigmoid", cv=5) for e in estimators_minmax],
        "feature_extractor": [CountVectorizer(), TfidfVectorizer()],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "min_df": [0, 1, 2],
        "scaler": [MinMaxScaler(feature_range=(0, 1))]
    }

    grid_scaled = {
        "estimator": [CalibratedClassifierCV(e, method="sigmoid", cv=5) for e in estimators_scaled],
        "feature_extractor": [CountVectorizer(), TfidfVectorizer()],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "min_df": [0, 1, 2],
        "scaler": [StandardScaler()]
    }

    # Perform grid search
    gs = GridSearchCV(NLPClassifier(lexicon=lexicon), param_grid=[
        grid_minmax,
        grid_scaled
    ], verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Pretty-print cv results
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(gs.cv_results_))

    print(gs.best_estimator_)

    print(f"Selected parameters: {gs.best_params_}")
    print(f"Mean score on validation set: {gs.best_score_}")
    print(
        f"Balanced accuracy score on test set of {ceil((1 - float(args.s)) * 100)}%: {gs.score(X_test, y_test)}")
    
    print(pd.DataFrame(classification_report(y_test, gs.best_estimator_.predict(X_test), output_dict=True)))

    # Retrain the model on full data
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)

    final_estimator = NLPClassifier(
        estimator=CalibratedClassifierCV(clone(gs.best_estimator_.estimator.base_estimator),
                                         method="sigmoid", cv=5, n_jobs=-1),
        feature_extractor=clone(gs.best_estimator_.feature_extractor),
        min_df=gs.best_estimator_.min_df,
        ngram_range=gs.best_estimator_.ngram_range,
        lexicon=lexicon,
        scaler=MinMaxScaler(feature_range=(0, 1))
    )

    final_estimator.fit(X_full, y_full)

    joblib.dump(final_estimator, "sentiment_classifier.joblib.pkl")

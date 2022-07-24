import nltk
from nltk.sentiment.util import mark_negation
import numpy as np
import string
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler

def lexicon_sentiment(tokenized_sentence, lexicon):
    """
    Extracts lexicon-based sentiment scored for a tokenized sentence.
    Parameters:
        - lexicon: lexicon to derive sentiment scores from. Should be a dictionary with format {"word": score}
        - tokenized_sentence: A tokenized sentence to calculate the total score for
    """

    # Return flipped score if the word is negated, otherwise return "real" score
    return sum([-int(lexicon.get(w, 0)) if "_NEG" in w else int(lexicon.get(w, 0)) for w in tokenized_sentence])

def treebank_to_wordnet(s, pos_tag):
    """
    Converts treebank word tags to wordnet tags
    Arguments:
        - s:        Untagged string
        - pos_tag:  Text string to convert
    """

    match pos_tag[0]:
        case 'J':
            return (s, 'a')
        case 'N':
            return (s, 'n')
        case 'V':
            return (s, 'v')
        case 'R':
            return (s, 'r')
        case _:
            return (s, '')


def tokenize_text(sentence):
    """ 
    Tokenizes a sentence:
        - Normalizes text to lowercase
        - Removes stopwords and punctuation
        - Tokenizes and lemmatizes wowrds
    """

    stopwords = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Get tokens
    toks = [w.lower() for w in nltk.tokenize.word_tokenize(sentence)]

    # POS options for wordnet
    pos = ['n', 'v', 'a', 'r']

    # Properly tag words for lemmatization
    lemmas = [lemmatizer.lemmatize(word, tag) if tag in pos else word for word, tag in nltk.tag.pos_tag(toks)]

    # Tag negations
    negations = mark_negation(lemmas, double_neg_flip=True)

    # Return filtered lemmas: stopwords and punctuation removed
    return [w for w in negations if w not in stopwords and w not in string.punctuation]

# Returns a dummy value. Required to disable default feature extractor preprocessors/tokenizers
def dummy(x):
    return x

class NLPClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, feature_extractor=CountVectorizer(), estimator=ComplementNB(), 
        ngram_range=(1, 1), min_df=0, lexicon=None, scaler=MinMaxScaler(feature_range=(0, 1))):
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
                - LogisticRegression()
            ngram_range: Range of ngrams to produce as features
            lexicon: Optional lexicon for extracting sentiment scores
            scaler: Sklearn scaler for scaling the data
        """

        self.feature_extractor = feature_extractor
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.lexicon = lexicon

        self.scaler = scaler

        self.estimator = estimator

    def fit(self, X, y):
        """
        Fits the the model.
        Arguments:
            X:          Input data
            y:          Input labels
        """

        # Get tokens
        tokens = [tokenize_text(s) for s in X.copy()]

        # Get feature matrix
        self.feature_extractor.set_params(preprocessor=dummy,
            tokenizer=dummy, ngram_range=self.ngram_range, min_df=self.min_df)
        X_processed = self.feature_extractor.fit_transform(tokens).toarray()

        # Get lexicon based features and append to matrix.
        # NOTE: Lexicon-based features cannot be used with Naive Bayes classifiers!
        if self.lexicon:
            if any([isinstance(self.estimator.base_estimator, e) for e in [ComplementNB, GaussianNB, MultinomialNB]]):
                self.lexicon = None
            else:
                X_processed = np.column_stack((X_processed, [lexicon_sentiment(t, self.lexicon) for t in tokens]))

        # Scale features
        if self.scaler:
            X_processed = self.scaler.fit_transform(X_processed)

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

        # Get tokens
        tokens = [tokenize_text(s) for s in X.copy()]

        # Get feature matrix
        X_processed = self.feature_extractor.transform(tokens).toarray()

        # Get lexicon based features and append to matrix
        if self.lexicon:
            X_processed = np.column_stack((X_processed, [lexicon_sentiment(t, self.lexicon) for t in tokens]))
        
        # Scale features
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)

        y = self.estimator.predict(X_processed)
        return y


    def predict_proba(self, X):
        """
        Generates probability predictions based on data.
        Arguments:
            X: (unseen) data
        """

        # Get tokens
        tokens = [tokenize_text(s) for s in X.copy()]

        # Get feature matrix
        X_processed = self.feature_extractor.transform(tokens).toarray()

        # Get lexicon based features and append to matrix
        if self.lexicon:
            X_processed = np.column_stack((X_processed, [lexicon_sentiment(t, self.lexicon) for t in tokens]))

        # Scale features
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)

        y = self.estimator.predict_proba(X_processed)
        return y
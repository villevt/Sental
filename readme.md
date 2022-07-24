# Sentimentai

## Summary

A small hobby project for practicing NLP sentiment analysis, and more so ML deployment. The intention is to create a simple text sentiment analyzer, slightly akin to tools such as Grammarly. The project will include:
* Command-line script for (re-)training supervised models based on grid search defined in the script itself (scikit-learn)
* Small full-stack webpage for demonstrating inference from the model (FastAPI, React)
* Deployment pipeline, for example to Heroku

![Screenshot of the application](/app.png?raw=true "Screenshot of the application")

## Custom NLP classifier

The NLP classifier is built using sklearn for the purpose of employing grid search over different models, by allowing the classifier itself to utilize different classifiers and feature extractors. The custom class is located in `nlpclassifier` and needs to be built by running `pip install .` in the corresponding folder. The custom model class is a requirement for both the backend, and model trainer utility.

## Command-line script

The command line script for model deployment is located in the `modeltrainer` folder. The script takes in text files in the following format for training

    negative text\t0\n
    positive text\t1\n
    ...
    
Where 0 denotes a negative, and 1 denotes a positive text.

The script then picks a best model out of multiple options, based on balanced accuracy score. Once the best model is selected, it is dumped into file `sentiment_classifier.joblib.pkl`.

The command line script expects the following syntax:

    python model_trainer.py
    -f  List of text files to use in training, relative to working or specified directory
    -l  A lexicon to use in training, relative to working or specified directory
    -d  Directory to search for text files, relative to working directory. Defaults to ./data
    -s  Size of training test, as fraction of total dataset    

## Modelling approach

The modelling approach here was relatively simple. I used my custom sklearn classifier with integrated text tokenization which (using NLTK):
- Normalizes text to lowercase
- Removes stopwords and punctuation
- Tokenizes and lemmatizes words

In addition to text tokenizations, I extracted lexicon-based monogram sentiment scores as features, using AFINN lexicon.

I represented the features with:
- Bag of words, 1 to 3 n-grams
- TF-IDF, 1 to 3 n-grams

I then employed an grid search over multiple different feature extractors and classifiers (to feed into my custom wrapper classifier), and chose the classifier with the best balanced accuracy (since I hadn't done a stratified train-test-split!). To get probabilities out of models that did not support sklearn's predict_proba-method, and also to calibrate other models, I used Platt's method for cross-validated probability calibration, as an "inner" cross validation. After finding the best model with this approach, I then trained the model on the entire dataset, also including test data, again with 5-fold calibration cross-validation to avoid overfitting.

Different classifiers used were:
- Complement Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Linear Support Vector Classifier
- Stochastic Gradient Descent Classifier
- (Lasso) Logistic Regression

Based on my results so far, it seems that Linear SVC is the best approach, when used with TF-IDF with mono- and bi-grams, and sentiment-based features derived from AFINN lexicon. The model exhibited a validation balanced accuracy of 82.11% on 1 800 observations' 5-fold CV, and a test balanced accuracy of 82.19% on a test set of 1 200 observations.

Note that the intention of this project is not to dive deep into optimizing the models and hyperparameter tuning - with careful adjustments, it could probably be possible to increase the test accuracy by a percentage point or two.

## Backend

The backend for the project is a minimalistic FastAPI server that responds to text requests with inferences (positive/negative). Following limitations are in place:
- Requests can only contain upwards to 400 characters
- A single IP address can only make up to 100 requests per day

Backend can be run from `backend`-folder with `uvicorn main:app`.

Since the backend is built on FastAPI, the backend also comes with Swagger documentation in path `/docs` in the api.

## Frontend

Minimal (TypeScript) React single page app, that allows for inputting text, and getting automatic inference for the text. The inference shows the sentiment and probability of the classification.

Frontend can be run from `frontend`-folder with `npm start`.

NOTE: The frontend is not responsive and won't look good on mobile devices.

## Model data

I constructed the model on sentiment data (`modeltrainer/data`) derived from https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set

The original dataset is attributed to _'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015_

The data consists of 500 positive-labeled (1) and 500 negative-lableled (0) reviews from Amazon, IMDB and Yelp each. The reviews and labels are tab-delimited, with observations delimited by linebreaks.

Obviously, the dataset is focused on reviews and might hence not be fully generalizable to communication in other contexts. Furthermore, the datasets are quite small. To better assess (and train) the generality of these models, data from different type of sources (e.g. literature, instant messaging) should be obtained. Additionally, it could be helpful to establish feedback loops to get more accurate predictions, for example with the following scheme
1. Users send their text for prediction, and receive a sentiment classification
2. The user can flag the classification as correct / incorrect, which is sent back to the system
3. The server does some filtering, e.g. to avoid spam/malicious flags¨
4. After filtering, the data is added to a "backlog"
5. Depending on the ML method being used, the "backlog" is used to either partially retrain the model each day, or the full model is retrained at specific time intervals with new entries added to the entire training set from the feedback loop

In addition to the labeled text data, I used the AFINN-111 Lexicon available from http://www2.imm.dtu.dk/pubdb/pubs/6010-full.html, which was originally used in _'Lars Kai Hansen, Adam Arvidsson, Finn Årup Nielsen, Elanor Colleoni,
Michael Etter, "Good Friends, Bad News - Affect and Virality in
Twitter", The 2011 International Workshop on Social Computing,
Network, and Services (SocialComNet 2011)._

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

The script then picks a best model out of multiple options, based on balanced accuracy score. Once the best model is selected, it is dumped into file `model.joblib`.

The command line script expects the following syntax:

    python model_trainer.py
    -f  List of text files to use in training, relative to working or specified directory
    -d  Directory to search for text files, if other than working directory
    -s  Size of training test, as fraction of total dataset    

## Modelling approach

The modelling approach here was relatively simple. I used my custom sklearn classifier with integrated text tokenization which (using NLTK):
- Normalizes text to lowercase
- Removes stopwords and punctuation
- Tokenizes and lemmatizes words

Having the custom wrapper classifier in place, which, I employed an grid search over multiple different feature extractors and classifiers (to feed into my custom wrapper classifier), and chose the classifier with the best balanced accuracy (since I hadn't done a stratified train-test-split!). Furthermore, to get probabilities out of models that did not support those, and also to calibrate other models I used a Platt's method for cross-validated probability calibration, as an "inner" cross validation. After finding the best model with this approach, I then trained the model on the entire dataset, also including test data, again with 5-fold calibration cross-validation to avoid overfitting.

I represented the features with:
- Bag of words, 1 to 3 n-grams
- TF-IDF, 1 to 3 n-grams

Different classifiers used were:
- Complement Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Linear Support Vector Classifier
- Stochastic Gradient Descent Classifier
- (Lasso) Logistic Regression

Based on my results so far, it seems that Linear SVC is the best approach, together with TF-IDF, and mono- and bi-gram features. The model exhibited a validation balanced accuracy of 80.30% on 1 800 observations' 5-fold CV, and a test balanced accuracy of 80.79% on a test set of 1 200 observations.

Note that we are only comparing different models here with default hyperparameters, and not getting deep into hyperparameter-tuning.

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

I constructed the model on sentiment data (`nlp/data`) derived from https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set

The original dataset is attributed to _'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015_

The data consists of 500 positive-labeled (1) and 500 negative-lableled (0) reviews from Amazon, IMDB and Yelp each. The reviews and labels are tab-delimited, with observations delimited by linebreaks.

Obviously, the dataset is focused on reviews and might hence not be fully generalizable to communication in other contexts. Furthermore, the datasets are quite small. To better assess (and train) the generality of these models, data from different type of sources (e.g. literature, instant messaging) should be obtained.

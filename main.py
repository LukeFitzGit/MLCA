#importing libraries
import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from training_tools import data_preprocessor


def main():
    #data preprocessing
    print('data preprocessing...')
    data_processor = data_preprocessor("data/combined_data.csv")
    data_processor.build_dataframe_from_csv(pd)
    data_processor.create_cleaned_data()
    data_df = data_processor.get_dataframe()

    #sampling data
    print('sampling...')
    sample_size = int(len(data_df) * 0.3)
    sample_df = data_df.sample(n=sample_size, random_state=42)
    #vectorizing and splitting testing and training data
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(sample_df.clean_text)
    y = sample_df.label
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    #svm
    print("fitting svm model...")
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)

    #logistic regression
    print("fitting logistic regression model...")
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    #naive bayes
    print("fitting naive bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)

    #interface
    inp = 0
    print('Would you like to test the prediction or enter an email in for prediction?')
    print('(1) test     (2) enter email     (anything else) exit')
    inp = input()
    if inp == '1':
        test_prediction(svm_model, lr_model, nb_model, x_test, y_test)
    if inp == '2':
        do_input_predictions(vectorizer, svm_model, lr_model, nb_model)
    else:
       exit()

#code for when predicted user inputs
def do_input_predictions(vectorizer, svm_model, lr_model, nb_model):
    x = ''
    while x != "exit":
        print('enter email or type "exit" to exit:')
        x = input()
        if x == 'exit':
            exit()
        result = do_prediction(x, vectorizer, svm_model, lr_model, nb_model)
        print("result: ", result)

#displays confusion matrix to screen
def display_confusion_matrix(plt, sns, confusion_matrix):
    #display confusion matrix using matplotlib
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

#use to get a result for when the user enters an email
def do_prediction(text: str, vectorizer, svm_model, lr_model, nb_model):
    text = do_text_preprocessing(text)
    vect_text = vectorizer.transform([text])
    svm_prediction = svm_model.predict(vect_text)
    lr_prediction = lr_model.predict(vect_text)
    nb_prediction = nb_model.predict(vect_text)
    combined_pred = np.argmax(np.bincount(np.hstack((svm_prediction, lr_prediction, nb_prediction))))
    return 'Spam' if combined_pred == 1 else 'Not Spam'

#preprocessing text
def do_text_preprocessing(text: str):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = text.strip()
    return text

#for when the user chooses to test model. predicts each model and returns individual and final results
def test_prediction(svm_model, lr_model, nb_model, x_test, y_test):
    print("predicting...")
    svm_pred = svm_model.predict(x_test)
    print("svm_classifiction report\n:", classification_report(y_test, svm_pred))
    lr_model_pred = lr_model.predict(x_test)
    print("lr_classifiction report\n:", classification_report(y_test, lr_model_pred))
    nb_model_pred = nb_model.predict(x_test)
    print("nb classification report\n:", classification_report(y_test, nb_model_pred))

    #combining results of each model
    combined_predictions = np.vstack((svm_pred, lr_model_pred, nb_model_pred)).T
    final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=combined_predictions)
    accuracy = accuracy_score(y_test, final_predictions)
    print("final accuracy: ", accuracy)
    print("would you like to display the confusion matrix?")
    inp = input('y/n')
    if inp.lower() == 'y':
        #confusion_matrix setup and display
        confusion_m = confusion_matrix(y_test, final_predictions)
        display_confusion_matrix(plt, sns, confusion_m)
    else:
        exit()


if __name__ == '__main__':
    main()

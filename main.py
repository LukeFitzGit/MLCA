#importing libraries
import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

def main():
    #getting csv data and converting into a pandas dataframe
    data_df = pd.read_csv("data/combined_data.csv")
    print('cleaning email data...')
    data_df['clean_text'] = data_df['text'].apply(preprocess_text)
    #splitting data into training and testing data
    x_training, x_testing, y_training, y_testing = train_test_split(data_df.clean_text, data_df.label)

    print('applying count vectorizer...')
    #text preprocessing, feature extraction & numerical representation/matrix creation
    cv = CountVectorizer()
    x_train_count_matrix = cv.fit_transform(x_training.values)

    print('fitting model...')
    #fitting the model
    model = MultinomialNB()
    model.fit(x_train_count_matrix, y_training)

    print('testing model...')
    #model test
    x_test_count_matrix = cv.transform(x_testing)
    print("Model score with just Naive Bayes:", model.score(x_test_count_matrix, y_testing))

    y_pred = model.predict(x_test_count_matrix)
    confusion_m = confusion_matrix(y_testing, y_pred)
    print('building chart...')

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_m, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = text.strip()
    return text

if __name__ == '__main__':
    main()

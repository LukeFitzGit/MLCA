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
from model import classification_model, data_preprocessor

def main():
#    model = classification_model()
    print('processing...')
    data_processor = data_preprocessor("data/combined_data.csv")
    data_processor.build_dataframe_from_csv(pd)
    data_processor.create_cleaned_data()
    data_df = data_processor.get_dataframe()
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(data_df.clean_text)
    y = data_df.label
    #getting csv data and converting into a pandas dataframe
   # data_df = pd.read_csv("data/combined_data.csv")
   # print('Descriptive Statistics:\n', data_df['text'].str.len().describe())
    #data_df['clean_text'] = data_df['text'].apply(preprocess_text)

    #splitting data into training and testing data
   # model = classification_model()
    x_training, x_testing, y_training, y_testing = train_test_split(x, data_df.label)

    #model.create_countVect_matrix()
    
    #text preprocessing, feature extraction & numerical representation/matrix creation
   # cv = CountVectorizer()
   # x_train_count_matrix = cv.fit_transform(x_training.values)

    #fitting the model
   # nb_model = MultinomialNB()
   # nb_model.fit(x_train_count_matrix, y_training)
   # nb_pred = nb_model.predict(x_testing)

    #model test
   # x_test_count_matrix = cv.transform(x_testing)
    #print("Model score with just Naive Bayes:", model.score(x_test_count_matrix, y_testing))
   # print("nb pred classification report\n:", classification_report(y_testing, nb_pred))

   #decision tree
   # try:
   #     print("fill na...")
   #     data_df.fillna(data_df.mean(), inplace=True)
   #     #decision tree
   #     print("vectorizer...")
   #     vectorizer = TfidfVectorizer()
   #     print("fit transform...")
   #     x = vectorizer.fit_transform(data_df.clean_text)
   #     y = data_df.label
   #     print("training...")
   #     x_train, x_test, y_train, y_test = train_test_split(x, y)
   #     print("dt classifier")
   #     dt_class = tree.DecisionTreeClassifier(max_depth=5)
   #     print("fitting...")
   #     dt_class.fit(x_train, y_train)
   #     print("decision tree:", dt_class.score(x_test, x_train))
   # except ex:
   #     print(ex)


    #svm
    print('sampling..')
    sample_size = int(len(data_df) * 0.3)
    sample_df = data_df.sample(n=sample_size, random_state=42)
    print("doing svm...")
    vectorizer = TfidfVectorizer()
    print("fit transform...")
    x = vectorizer.fit_transform(sample_df.clean_text)
    y = sample_df.label
    print("train split")
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print("svm model fit...")
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)

    #logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    #naive bayes
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)

    #test_prediction(svm_model, lr_model, nb_model, x_test, y_test)

    x = ''
    while x != "exit":
        print('enter email:')
        x = input()
        result = do_prediction(x, vectorizer, svm_model, lr_model, nb_model)
        print("result: ", result)

    #confusion_matrix setup
    #y_pred = model.predict(x_test_count_matrix)
    #confusion_m = confusion_matrix(y_testing, y_pred)

    #classification report
    #print(classification_report(y_testing, y_pred))

    #code to display confusion matrix
    #display_confusion_matrix(plt, sns, confusion_m)



def do_text_preprocessing(text: str):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = text.strip()
    return text


def display_confusion_matrix(plt, sns, confusion_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def do_prediction(text: str, vectorizer, svm_model, lr_model, nb_model):
    text = do_text_preprocessing(text)
    vect_text = vectorizer.transform([text])
    svm_prediction = svm_model.predict(vect_text)
    lr_prediction = lr_model.predict(vect_text)
    nb_prediction = nb_model.predict(vect_text)

    combined_pred = np.argmax(np.bincount(np.hstack((svm_prediction, lr_prediction, nb_prediction))))
    return 'Spam' if combined_pred == 1 else 'Not Spam'



def test_prediction(svm_model, lr_model, nb_model, x_test, y_test):
    print("predicting...")
    svm_pred = svm_model.predict(x_test)
    print("svm_pred:\n", svm_pred)
#    print("score:\n", svm_model.score(x_test, x_train))
    svm_acc = accuracy_score(y_test, svm_pred)
    print('accuracy: ', svm_acc) 
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


    #confusion_matrix setup
    confusion_m = confusion_matrix(y_test, final_predictions)

    #code to display confusion matrix
    #display_confusion_matrix(plt, sns, confusion_m)






if __name__ == '__main__':
    main()

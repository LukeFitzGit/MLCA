import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

class classification_model:
    def run_naive_bayes(self):
        return 0

    def get_naive_bayes_result(self):
        return 0

    def run_svm(self):
        return 0

    def get_svm_result(self):
        return 0

    def create_countVect_matrix(self):
        cv = CountVectorizer()
        self._x_train_count_matrix = cv.fit_transform(self._x_training.values)


class data_preprocessor:
    def __init__(self, csv_path):
        self._csvpath = csv_path


    def build_dataframe_from_csv(self, pandas):
        self._data_df = pandas.read_csv(self._csvpath)

    def get_dataframe(self):
        return self._data_df

    def create_cleaned_data(self):
        self._data_df['clean_text'] = self._data_df['text'].apply(data_preprocessor.preprocess_text)


    def preprocess_text(text: str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        text = text.strip()
        return text



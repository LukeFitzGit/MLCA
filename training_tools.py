import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

#class or preproccessing data
class data_preprocessor:
    def __init__(self, csv_path):
        self._csvpath = csv_path

    #build dataframe
    def build_dataframe_from_csv(self, pandas):
        self._data_df = pandas.read_csv(self._csvpath)

    #get dataframe
    def get_dataframe(self):
        return self._data_df

    #create cleaned data column
    def create_cleaned_data(self):
        self._data_df['clean_text'] = self._data_df['text'].apply(data_preprocessor.preprocess_text)

    #format and process email text
    def preprocess_text(text: str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        text = text.strip()
        return text

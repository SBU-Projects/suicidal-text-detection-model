import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



import re
import string

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import SelectFromModel

from imblearn.pipeline import Pipeline
import pickle


import spacy
nlp = spacy.load("en_core_web_lg")

import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("Datasets/cleaned.csv")

"""
from DataPreprocessing import DP
df = pd.read_csv("Datasets/Suicide_Ideation_Dataset(Twitter-based).csv")
dp = DP(df)
df = df[~df['Tweet'].apply(lambda x: isinstance(x, float))]
df['cleaned_text'] = df['Tweet'].apply(lambda row: dp.text_preprocessing(row))
df.to_csv('Datasets/cleaned.csv', index=False)
"""




class TextPreprocessor(TransformerMixin):
    def __init__(self, text_attribute):
        self.text_attribute = text_attribute

    def transform(self, X, *_):
        X_copy = X.copy()
        X_copy[self.text_attribute] = X_copy[self.text_attribute].apply(self._preprocess_text)
        return X_copy

    def _preprocess_text(self, text):
        return self._lemmatize(self._leave_letters_only(self._clean(text)))

    def _clean(self, text):
        bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'
        text_without_symbols = text.translate(str.maketrans('', '', bad_symbols))

        text_without_bad_words = ''
        for line in text_without_symbols.split('\n'):
            if not line.lower().startswith('from:') and not line.lower().endswith('writes:'):
                text_without_bad_words += line + '\n'

        clean_text = text_without_bad_words
        email_regex = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        regexes_to_remove = [email_regex, r'Subject:', r'Re:']
        for r in regexes_to_remove:
            clean_text = re.sub(r, '', clean_text)

        return clean_text

    def _leave_letters_only(self, text):
        text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))

    def _lemmatize(self, text):
        doc = nlp(text)
        words = [x.lemma_ for x in [y for y in doc if not y.is_stop and y.pos_ != 'PUNCT'
                                    and y.pos_ != 'PART' and y.pos_ != 'X']]
        return ' '.join(words)

    def fit(self, *_):
        return self

text_preprocessor = TextPreprocessor(text_attribute='Tweet')
df_preprocessed = text_preprocessor.transform(df)

train, test = train_test_split(df_preprocessed, test_size=0.3)

#Vectorize data
tfidf_vectorizer = TfidfVectorizer(analyzer = "word", max_features=10000)
X_tfidf_train = tfidf_vectorizer.fit_transform(train['Tweet'])
X_tfidf_test = tfidf_vectorizer.transform(test['Tweet'])

y = train['Suicide']
y_test = test['Suicide']

X, y = X_tfidf_train, y
X_test, y_test = X_tfidf_test, y_test

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X.toarray())
X_test_norm = scaler.transform(X_test.toarray())

lsvc = LinearSVC(C=100, penalty='l1', max_iter=500, dual=False)
lsvc.fit(X_norm, y)
fs = SelectFromModel(lsvc, prefit=True)
X_sel = fs.transform(X_norm)
X_test_sel = fs.transform(X_test_norm)
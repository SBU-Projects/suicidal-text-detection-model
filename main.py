import pandas as pd
from DataPreprocessing import DP

# Load dataset
df = pd.read_csv('Datasets/Suicide_Ideation_Dataset(Twitter-based).csv')
df.reset_index(drop=True, inplace=True)
""".
df['Suicide'] = df['Suicide'].replace({'Potential Suicide post ': 1, 'Not Suicide post': 0})
print(df)
.



"""
dp = DP(df)

df['cleaned_text'] = df['Tweet'][:20].apply(lambda row: dp.text_preprocessing(row))
# Convert all values to strings and handle missing values
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')



import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
import seaborn as sns

# Obtain word frequency
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
word_freq = pd.DataFrame(tokenizer.word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)

# Remove rows with text length 0
cleaned_df = df[df['cleaned_text'].apply(lambda x: len(x.split()) != 0)]
cleaned_df.reset_index(drop=True, inplace=True)
print(cleaned_df.head())

# Export cleaned dataset
cleaned_df.to_csv('Datasets/suicide_detection_final_cleaned.csv', index=False)


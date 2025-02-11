import pandas as pd
from DataPreprocessing import DP

df = pd.read_csv("Datasets/suicide_detection_final_cleaned.csv")
dp = DP(df)
df = df[~df['Tweet'].apply(lambda x: isinstance(x, float))]
df['cleaned_text'] = df['Tweet'].apply(lambda row: dp.text_preprocessing(row))

print(df.head())
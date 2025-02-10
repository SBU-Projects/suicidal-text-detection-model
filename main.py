import pandas as pd
import spacy

# Load dataset
df = pd.read_csv('Datasets/Suicide_Ideation_Dataset(Twitter-based).csv')
df.reset_index(drop=True, inplace=True)

df['Suicide'] = df['Suicide'].replace({'Potential Suicide post ': 1, 'Not Suicide post': 0})
print(df)
import pandas as pd

# Load dataset
df = pd.read_csv('Datasets/Suicide_Ideation_Dataset(Twitter-based).csv')
df.reset_index(drop=True, inplace=True)

print(df.head())
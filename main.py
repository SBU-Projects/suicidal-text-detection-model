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



import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
import seaborn as sns

# Obtain word frequency
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'][:100])
word_freq = pd.DataFrame(tokenizer.word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)


# Plot bar graph for word frequency
plt.figure(figsize=(16, 8))
sns.barplot(x='count',y='word',data=word_freq.iloc[:100])
plt.title('Most Frequent Words')
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

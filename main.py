import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from DataPreprocessing import DP

df = pd.read_csv("Datasets/suicide_detection_final_cleaned.csv")
dp = DP(df)
df = df[~df['Tweet'].apply(lambda x: isinstance(x, float))]
df['cleaned_text'] = df['Tweet'][:100].apply(lambda row: dp.text_preprocessing(row))

comment_words = ''
stopwords = set(STOPWORDS)

# iterate through the csv file
for val in df['cleaned_text'][:100].CONTENT:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
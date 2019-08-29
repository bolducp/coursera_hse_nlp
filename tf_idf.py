from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import nltk
nltk.download('punkt')

o_henry_story = open("the_gift_of_the_magi.txt", "r").read()
sentences = nltk.sent_tokenize(o_henry_story)[:15]

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf.fit_transform(sentences)
df = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)

print(df)

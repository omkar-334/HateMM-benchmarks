import gensim.downloader as api
import numpy as np
import pandas as pd

df = pd.read_csv("transcriptions.csv")

glove_model = api.load("glove-wiki-gigaword-100")


def text_to_embedding(text):
    words = text.lower().split()
    word_vecs = [glove_model[word] for word in words if word in glove_model]

    if word_vecs:
        return np.mean(word_vecs, axis=0)
    else:
        return np.zeros(100)


df["features"] = df["text"].apply(text_to_embedding)

df.to_pickle("text_features.pkl")

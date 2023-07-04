import csv
import random
import re
import string

import gdown
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

#FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP



# 学習済み単語ベクトルのダウンロード
# url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
# output = 'GoogleNews-vectors-negative300.bin.gz'
# gdown.download(url, output, quiet=True)

#70. 
filename = "NewsAggregatorDataset/newsCorpora.csv"
df = pd.read_csv(filename, header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df = df.loc[df["PUBLISHER"].isin( ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])]

train, test_valid_data = train_test_split(df, test_size=0.2, shuffle=True)
test, valid = train_test_split(test_valid_data, test_size=0.5, shuffle=True)

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

def create_corpus(df):
    corpus = df.TITLE.values.tolist()
    for i in range(len(corpus)):
        text = corpus[i].replace("...", "")
        text = re.sub(r"[!\'\"()*+,-./:;<=>?@]", "", text)
        text = re.sub("\d+", "0", text)
        corpus[i] = text.strip()
    return corpus

def transform_t2v(text):
    words = text.split(" ")
    vec = [model[word] for word in words if word in model]
    return torch.tensor(sum(vec) / len(vec))

#70. 
train_corpus = create_corpus(train)
test_corpus = create_corpus(test)
valid_corpus = create_corpus(valid)

X_train = torch.stack([transform_t2v(text) for text in train_corpus])
X_valid = torch.stack([transform_t2v(text) for text in valid_corpus])
X_test = torch.stack([transform_t2v(text) for text in test_corpus])

category_label = {"b": 0, "t": 1, "e": 2, "m": 3}
y_train = torch.tensor(list(map(lambda x: category_label[x], train["CATEGORY"])))
y_test = torch.tensor(list(map(lambda x: category_label[x], test["CATEGORY"])))
y_valid = torch.tensor(list(map(lambda x: category_label[x], valid["CATEGORY"])))

torch.save(X_train, 'result/X_train.pt')
torch.save(X_valid, 'result/X_valid.pt')
torch.save(X_test, 'result/X_test.pt')
torch.save(y_train, 'result/y_train.pt')
torch.save(y_valid, 'result/y_valid.pt')
torch.save(y_test, 'result/y_test.pt')
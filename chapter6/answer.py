import csv
import random
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
#FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

#50. JSONデータの読み込み
filename = "NewsAggregatorDataset/newsCorpora.csv"
df = pd.read_csv(filename, header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df = df.loc[df["PUBLISHER"].isin( ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])]

train, test_valid_data = train_test_split(df, test_size=0.2, shuffle=True)
test, valid = train_test_split(test_valid_data, test_size=0.5, shuffle=True)

#保存
train.to_csv("train.txt", sep="\t", index=False)
valid.to_csv("valid.txt", sep="\t", index=False)
test.to_csv("test.txt", sep="\t", index=False)


#51
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression

vectorizer = TfidfVectorizer(max_df=0.9)
X_train = vectorizer.fit_transform(train["TITLE"])
X_valid = vectorizer.transform(valid["TITLE"])
X_test = vectorizer.transform(test["TITLE"])


X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
X_valid = pd.DataFrame(X_valid.toarray(), columns=vectorizer.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())


# データの保存
X_train.to_csv('X_train.txt', sep='\t', index=False)
X_valid.to_csv('X_valid.txt', sep='\t', index=False)
X_test.to_csv('X_test.txt', sep='\t', index=False)

#52
y_train = train["CATEGORY"]
y_test = test["CATEGORY"]

model = LogisticRegression(penalty="l2")
model.fit(X_train, y_train)

#53
train_pred = [np.max(model.predict_proba(X_train), axis=1), model.predict(X_train)]
test_pred = [np.max(model.predict_proba(X_test), axis=1), model.predict(X_test)]

print(train_pred)
print(test_pred)


#54
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

train_pred = train_pred[1]
test_pred = train_pred[1]

print("train")
print("accuracy = ", accuracy_score(y_true=y_train, y_pred=train_pred))

print("test")
print("accuracy = ", accuracy_score(y_true=y_test, y_pred=test_pred))

#55
# 混合行列
print(confusion_matrix(y_true=y_train, y_pred=train_pred))
print(confusion_matrix(y_true=y_test, y_pred=test_pred))
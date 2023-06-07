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
y_valid = valid["CATEGORY"]

model = LogisticRegression(penalty="l2")
model.fit(X_train, y_train)

#53
train_pred = [np.max(model.predict_proba(X_train), axis=1), model.predict(X_train)]
test_pred = [np.max(model.predict_proba(X_test), axis=1), model.predict(X_test)]

# print(train_pred)
# print(test_pred)


#54
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

train_pred = train_pred[1]
test_pred = test_pred[1]

# print("train")
# print("accuracy = ", accuracy_score(y_true=y_train, y_pred=train_pred))

# print("test")
# print("accuracy = ", accuracy_score(y_true=y_test, y_pred=test_pred))

#55
# 混合行列
print(confusion_matrix(y_true=y_train, y_pred=train_pred))
print(confusion_matrix(y_true=y_test, y_pred=test_pred))


#56

# 適合率
precision = precision_score(y_true=y_test, y_pred=test_pred, average=None, labels=["b", "e", "t", "m"])
precision_micro = precision_score(y_true=y_test, y_pred=test_pred, average="micro")
precision_macro = precision_score(y_true=y_test, y_pred=test_pred, average="macro")
precision = np.append(precision, [precision_micro, precision_macro])

# 再現率
recall = recall_score(y_true=y_test, y_pred=test_pred, average=None, labels=["b", "e", "t", "m"])
recall_micro = recall_score(y_true=y_test, y_pred=test_pred, average="micro")
recall_macro = recall_score(y_true=y_test, y_pred=test_pred, average="macro")
recall = np.append(recall, [recall_micro, recall_macro])


# f1スコア
f1 = f1_score(y_true=y_test, y_pred=test_pred, average=None, labels=["b", "e", "t", "m"])
f1_micro = f1_score(y_true=y_test, y_pred=test_pred, average="micro")
f1_macro = f1_score(y_true=y_test, y_pred=test_pred, average="macro")
f1 = np.append(f1, [f1_micro, f1_macro])

scores = pd.DataFrame({"precision" : precision, "recall" : recall, "f1_score" : f1}, index=["b", "e", "t", "m", "micro", "macro"])
print(scores, end="\n\n")


#57
features = X_train.columns.values
for c, coef in zip(model.classes_, model.coef_):
    print(f"Categolly : {c}")
    
    best10 = features[np.argsort(coef)[::-1][:10]]
    print("Best : ", end="")
    for num in range(10):
        print(best10[num], end=" ")
    print("")
    worst10 = features[np.argsort(coef)[:10]]  
    print("Worst : ", end="")
    for num in range(10):
        print(worst10[num], end=" ")
    print("\n")

#58
from tqdm import tqdm
import matplotlib.pyplot as plt

def pred_score(model, data):
    pred = [np.max(model.predict_proba(data), axis=1), model.predict(data)]
    return pred

# result = []
# for C in tqdm(np.logspace(-5, 4, 10, base=10)):
#     model = LogisticRegression(random_state=42, C=C)
#     model.fit(X_train, y_train)

#     # 予測値の取得
#     train_pred = pred_score(model, X_train)[1]
#     valid_pred = pred_score(model, X_valid)[1]
#     test_pred = pred_score(model, X_test)[1]

#     # 正解率の算出
#     train_accuracy = accuracy_score(y_train, train_pred)
#     valid_accuracy = accuracy_score(y_valid, valid_pred)
#     test_accuracy = accuracy_score(y_test, test_pred)

#     # 結果の格納
#     result.append([C, train_accuracy, valid_accuracy, test_accuracy])

# result = np.array(result).T
# plt.plot(result[0], result[1], label="train")
# plt.plot(result[0], result[2], label="valid")
# plt.plot(result[0], result[3], label="test")
# plt.ylim(0, 1.1)
# plt.ylabel("Accuracy")
# plt.xscale("log")
# plt.xlabel("C")
# plt.legend()
# plt.show()

#59
result = []
for l1_ratio in tqdm(np.linspace(0, 1, num=10)):
    model = LogisticRegression(random_state=42, solver="saga", penalty="elasticnet", l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    # 予測値の取得
    train_pred = pred_score(model, X_train)[1]
    valid_pred = pred_score(model, X_valid)[1]
    test_pred = pred_score(model, X_test)[1]

    # 正解率の算出
    train_accuracy = accuracy_score(y_train, train_pred)
    valid_accuracy = accuracy_score(y_valid, valid_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # 結果の格納
    result.append([l1_ratio, train_accuracy, valid_accuracy, test_accuracy])

result = np.array(result).T
plt.plot(result[0], result[1], label="train")
plt.plot(result[0], result[2], label="valid")
plt.plot(result[0], result[3], label="test")
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.xlabel("l1_ratio")
plt.legend()
plt.show()

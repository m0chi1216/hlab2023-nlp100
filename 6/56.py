import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


label={'b':0,'e':1,'t':2,'m':3}

def processing(data):
    x=[]
    y=[]
    for title,categoly in data:
        title=re.sub('[0-9]+','0',title)
        x.append(title.lower())
        y.append(label[categoly])
    return x,y

def pre_score(lg,X):
    pre=lg.predict([X])
    pre_prob=lg.predict_proba([X])[0,pre]
    return pre[0],pre_prob[0]

# データの読込
df=pd.read_csv('NewsAggregatorDataset/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df = df[['TITLE','CATEGORY']]

#ret = train_test_split(arrays, [test_size], [train_size], [random_state], [shuffle], [stratify])
#分割はlist、ndarray、DataFrame

train,valid_once=train_test_split(df,test_size=0.2,shuffle=True,random_state=100,stratify=df['CATEGORY'])
valid,test=train_test_split(valid_once,test_size=0.5,shuffle=True,random_state=100,stratify=valid_once['CATEGORY'])

train = np.array(train)
valid = np.array(valid)
test  = np.array(test)

x_train, y_train = processing(train)
x_valid, y_valid = processing(valid)
x_test , y_test  = processing(test)

#print(train)
#a=np.array([[1,2],[3,4]])
#print(a)

tfidf=TfidfVectorizer(min_df=10)
tfidf.fit(x_train)
x_train = tfidf.transform(x_train).toarray()
x_valid = tfidf.transform(x_valid).toarray()
x_test = tfidf.transform(x_test).toarray()
#fitでidf,transformでtf

from sklearn.linear_model import LogisticRegression

# モデルの学習
lg = LogisticRegression(random_state=100, max_iter=200)
lg.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
pre_train =[]
pre_test=[]

for i in x_train:
    pre_train.append(lg.predict([i]))
for i in x_test:
    pre_test.append(lg.predict([i]))

'''
train_accuracy = accuracy_score(y_train,pre_train)
test_accuracy = accuracy_score(y_test,pre_test)
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')
'''
'''
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_matrix=confusion_matrix(y_train,pre_train)
print(train_matrix)
print(type(train_matrix))
sns.heatmap(train_matrix, annot=True, cmap='Blues')
plt.show()
'''

from sklearn.metrics import precision_score, recall_score, f1_score

def calk(Y,pred):
    precision = precision_score(Y, pred,average=None)
    precision = np.append(precision,precision_score(Y, pred, average='micro'))
    precision = np.append(precision,precision_score(Y, pred, average='macro'))
    #precision_mi = precision_score(Y, pred, average='micro').reshape(1)
    #precision_ma = precision_score(Y, pred, average='macro') .reshape(1)
    #precision =np.concatenate([precision,precision_mi,precision_ma])

    recall = recall_score(Y, pred,average=None)
    recall = np.append(recall,recall_score(Y, pred, average='micro'))
    recall = np.append(recall,recall_score(Y, pred, average='macro'))

    f1 = f1_score(Y, pred,average=None)
    f1 = np.append(f1,f1_score(Y, pred, average='micro'))
    f1 = np.append(f1,f1_score(Y, pred, average='macro'))

    score = pd.DataFrame({'適合率':precision,'再現率':recall,'F1スコア':f1},index=['b','e','t','m','micro','macro'])
    #マクロ平均法を採用すると、各クラスのサンプル数の偏りに影響を受けることなく評価指標が算出できる
    #マイクロ平均法はデータセットのサンプル数全体を考慮して評価指標を算出する方法
    return score

print(calk(y_train,pre_train))
import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def processing(data):
    x=[]
    y=[]
    for title,categoly in data:
        title=re.sub('[0-9]+','0',title)
        x.append(title.lower())
        y.append(categoly)
    return x,y

def pre_score(lg,X):
    pre=lg.predict([X])
    pre_prob=lg.predict_proba([X])[0,pre]
    return pre[0],pre_prob[0]

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

# データの読込
df=pd.read_csv('NewsAggregatorDataset/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df = df[['TITLE','CATEGORY']]

#ret = train_test_split(arrays, [test_size], [train_size], [random_state], [shuffle], [stratify])
#分割はlist、ndarray、DataFrame

train,valid_once=train_test_split(df,test_size=0.2,shuffle=True,random_state=123,stratify=df['CATEGORY'])
valid,test=train_test_split(valid_once,test_size=0.5,shuffle=True,random_state=123,stratify=valid_once['CATEGORY'])

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

#x_train = pd.DataFrame(x_train,columns=tfidf.get_feature_names_out())
#x_valid = pd.DataFrame(x_valid,columns=tfidf.get_feature_names_out())
#x_test = pd.DataFrame(x_test,columns=tfidf.get_feature_names_out())

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



label={0:'b',1:'e',2:'t',3:'m'}
name = tfidf.get_feature_names_out()
for i,coef in enumerate(lg.coef_):
    top = name[np.argsort(coef)][::-1][:10]
    worst = name[np.argsort(coef)][:10]
    #wrost = name[np.argsort(coef)]
    #print(np.sort(coef))
    print(label[i])
    print('top',top)
    print('worst',worst)


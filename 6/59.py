import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import optuna

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


import optuna



# 最適化対象を関数で指定
def objective_lg(trial):
  # チューニング対象パラメータのセット
  l1_ratio = trial.suggest_uniform('l1_ratio', 0, 0.7)
  C = trial.suggest_loguniform('C', 1e-3, 1e3)

  # モデルの学習
  lg = LogisticRegression(random_state=100, 
                          max_iter=200, 
                          penalty='elasticnet', 
                          solver='saga', 
                          l1_ratio=l1_ratio, 
                          C=C)
  lg.fit(x_train, y_train)

  # 予測値の取得
  valid_pred = lg.predict(x_valid)

  # 正解率の算出
  valid_accuracy = accuracy_score(y_valid, valid_pred)    

  return valid_accuracy 
# 最適化
study = optuna.create_study(direction='maximize')
study.optimize(objective_lg, timeout=30)

# 結果の表示
print('Best trial:')
trial = study.best_trial
print('  Value: {:.3f}'.format(trial.value))
print('  Params: ')



l1_ratio = trial.params['l1_ratio']
C = trial.params['C']
for key, value in trial.params.items():
  print('    {}: {}'.format(key, value))

# モデルの学習
lg = LogisticRegression(random_state=100, 
                        max_iter=200, 
                        penalty='elasticnet', 
                        solver='saga', 
                        l1_ratio=l1_ratio, 
                        C=C)
lg.fit(x_train, y_train)
pre_train =[]
pre_valid=[]
pre_test=[]

for i in x_train:
    pre_train.append(lg.predict([i]))
for i in x_valid:
    pre_valid.append(lg.predict([i]))
for i in x_test:
    pre_test.append(lg.predict([i]))

train_accuracy = accuracy_score(y_train,pre_train)
valid_accuracy = accuracy_score(y_valid,pre_valid)
test_accuracy = accuracy_score(y_test,pre_test)
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（開発データ）：{valid_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')
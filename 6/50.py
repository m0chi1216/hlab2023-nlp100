import json
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split
'''
datalist=[]
with open('NewsAggregatorDataset/newsCorpora.csv') as f:
  for line in f:
    pattern = r'^(.*(?:Reuters|Huffington Post|Businessweek|Contactmusic\.com|Daily Mail).*)$'
    if re.search(pattern,line):
        datalist.append(line)
random.shuffle(datalist)
print(''.join(datalist))
'''

#from sklearn.model_selection import train_test_split

# データの読込
df=pd.read_csv('NewsAggregatorDataset/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
#df[df['PUBLISHER']  == 'Reuters']

df = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
df = df[['TITLE','CATEGORY']]

#ret = train_test_split(arrays, [test_size], [train_size], [random_state], [shuffle], [stratify])
#分割はlist、ndarray、DataFrame

train,valid_once=train_test_split(df,test_size=0.2,shuffle=True,random_state=100,stratify=df['CATEGORY'])
#stratifyはどの要素の比率を同じにするか決める
#random_stateは？
valid,test=train_test_split(valid_once,test_size=0.5,shuffle=True,random_state=100,stratify=valid_once['CATEGORY'])

train.to_csv('./train.txt', sep='\t', index=False)
valid.to_csv('./valid.txt', sep='\t', index=False)
test.to_csv('./test.txt', sep='\t', index=False)


print(train['CATEGORY'].value_counts())
print(valid['CATEGORY'].value_counts())
print(test['CATEGORY'].value_counts())
#カテゴリーの比率を出力
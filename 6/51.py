import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def processing(data):
    x=[]
    y=[]
    label={'b':0,'e':1,'t':2,'m':3}
    for title,categoly in data:
        #table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        #title = title.translate(table)  # 記号をスペースに置換
        title=re.sub('[0-9]+','0',title)
        x.append(title.lower())
        y.append(label[categoly])
    return x,y

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

#うまくいかないやつ
tfidf=TfidfVectorizer(min_df=10)

tfidf.fit(x_train)
x_train = tfidf.transform(x_train)
x_valid = tfidf.transform(x_valid)
x_test = tfidf.transform(x_test)

x_train = pd.DataFrame(x_train.toarray(),columns=tfidf.get_feature_names_out())
x_valid = pd.DataFrame(x_valid.toarray(),columns=tfidf.get_feature_names_out())
x_test = pd.DataFrame(x_test.toarray(),columns=tfidf.get_feature_names_out())

'''
tfidf=TfidfVectorizer(min_df=0.01)
#mindfはいるのか
x_train = tfidf.fit_transform(x_train)
x_train = pd.DataFrame(x_train.toarray(),columns=tfidf.get_feature_names_out())
x_valid = tfidf.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid.toarray(),columns=tfidf.get_feature_names_out())
x_test = tfidf.fit_transform(x_test)
x_test = pd.DataFrame(x_test.toarray(),columns=tfidf.get_feature_names_out())
#fitでidf,transformでtf
'''

x_train.to_csv('./X_train.txt', sep='\t', index=False)
x_valid.to_csv('./X_valid.txt', sep='\t', index=False)
x_test.to_csv('./X_test.txt', sep='\t', index=False)
print('complete')

#print(tfidf.get_feature_names_out())
#x_train = pd.DataFrame(x_train,columns=tfidf.get_feature_names_out())

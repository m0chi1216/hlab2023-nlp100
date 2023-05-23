import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

pattern = r'^\{\{基礎情報.*?$(.*)^\}\}'
find = re.findall(pattern,text,re.MULTILINE+ re.DOTALL)
#print(find)
#re.dotallを使うことによって.のコマンドが改行にも対応するようになる

pattern = r'^\|(.*?)\s*=(.*?)$'
find2 = re.findall(pattern,find[0],re.MULTILINE)
dictlist = dict(find2)
'''
for i in dictlist:
    print('{} = {}'.format(i,dictlist[i]))
'''
for n,m in dictlist.items():
    print(f'{n} = {m}')

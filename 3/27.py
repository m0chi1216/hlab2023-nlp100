import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

pattern = r'^\{\{基礎情報.*?$(.*)^\}\}'
find = re.findall(pattern,text,re.MULTILINE+ re.DOTALL)
pattern = r'^\|(.*?)\s*= (.*?)$'
find2 = re.findall(pattern,find[0],re.MULTILINE)

dictlist={}
for i in range(len(find2)):
  #強調
  pattern = r'\'{2,5}'
  j = re.sub(pattern,'',find2[i][1])
  #ファイル
  pattern = r'\[\[ファイル.*\|([^\]]*?)\]\]'
  j = re.sub(pattern, r'\1', j)
  #内部
  pattern = r'\[\[(?:[^\]]*?\|)?([^\]]*)\]\]'
  j = re.sub(pattern, r'\1', j)

  dictlist[find2[i][0]]=j




for n,m in dictlist.items():
    print(f'{n} = {m}')



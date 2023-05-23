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

pattern = r'\'{2,5}'
dictlist={}
for i in range(len(find2)):
  dictlist[find2[i][0]]=re.sub(pattern,'',find2[i][1])



for n,m in dictlist.items():
    print(f'{n} = {m}')

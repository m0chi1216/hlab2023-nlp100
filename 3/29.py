import json
import re
import requests

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

  dictlist[find2[i][0]]=j


url=dictlist['国旗画像']
url=url.replace(' ','_')
wikiurl='https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url + '&prop=imageinfo&iiprop=url&format=json'
data = requests.get(wikiurl)
#print(data.text)
ans = re.search(r'"url":"(.+?)"', data.text).group(1)
print(ans)


import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

print('method1')
splittext=text.split('\n')
for li in splittext:
  if '[[Category:' in li:
    print(li)

#正規表現を扱う場合には文字列の先頭にrをつける
print('medthod2')
pattern = r'\[\[Category:.*\]\]'
for li in splittext:
    if re.fullmatch(pattern,li) !=None:
      print(li)

print('method3')
pattern2 = r'^\[\[Category:.*\]\]$'
find = re.findall(pattern2,text,re.MULTILINE)
print('\n'.join(find))
#Mulutilineの指定によって改行を考えて^や$のマッチングを行う→その後リスト化
#Mulutilineは基本的に^,$の使い方を変えるために使用と考えていい？
#今回は行を抜き出すが、^$の指定によって行と完全一致することが条件となる


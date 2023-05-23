import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

pattern2 = r'^\[\[Category:(.*?)(?:\|.*)?\]\]$'
find = re.findall(pattern2,text,re.MULTILINE)
print('\n'.join(find))

#findallでグループ表現を使うとグループの部分だけを抜き取る
#(?: )で始まることによって、グループ化しても拾わなくなる
#全部につければグループを使っていても全体を取得できる

#正規表現.*は貪欲マッチ(長さが最長になるようにマッチ).*?は非貪欲マッチ(文字列の長さが最小になるようにマッチ)
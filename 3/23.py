import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

pattern = r'＾=+(?:\s*)(.*?)(?:\s*)(=+)'
find = re.findall(pattern,text,re.MULTILINE)
for i in range(len(find)):
  print('{} {}'.format(find[i][0],len(find[i][1])-1))


#pattern = r'^(\={2,})\s*(.+?)\s*(\={2,}).*$'
#{2,}でくり返しの最低回数を指定することができる
import json
import re

with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      text = content['text']

pattern = r'\[\[(?:ファイル|File):((?:.*?)\.(?:.*?))\|'
find = re.findall(pattern,text)
print('\n'.join(find))

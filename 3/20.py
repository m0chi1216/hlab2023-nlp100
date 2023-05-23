import json


with open('jawiki-country.json') as f:
  for line in f:
    content = json.loads(line)
    if content['title'] == 'イギリス':
      print(content['text'])
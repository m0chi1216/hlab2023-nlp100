import json 

#20. JSONデータの読み込み
filename = 'jawiki-country.json'
with open(filename, mode='r', encoding="utf-8") as f:
  for line in f:
    line = json.loads(line)
    if line['title'] == 'イギリス':
      uk_text = line['text']
      break

#print(uk_text)

#21
import re
categories = re.findall(r'\[\[Category:.*\]\]', uk_text)
#print("\n".join(categories))

#22
categories_inside = re.findall(r"\[\[Category:(.*)\]\]", uk_text)
categories_inside = [category.split("|")[0] for category in categories_inside]
#print("\n".join(categories_inside))

#23
sections = re.findall(r"(\={2,})\s*(.+?)\s*(\={2,})", uk_text)
sections_result = "\n".join((section[1] + ":" + str(len(section[0]) - 1) for section in sections))
#print(sections_result)

#24
medias = re.findall(r"\[\[ファイル:(.+?)\|", uk_text)
#print("\n".join(medias))

#25
template = re.findall(r"\{\{基礎情報.*[\s\S]*\}\}", uk_text)
template = re.findall(r"\|(.*?)\s=\s(.*)", template[0])
result = dict(template)
# for k, v in result.items():
#   print(k + ": " + v)

#26
result_rm = {k : re.sub(r"\'{2,5}", "", v) for k,v in result.items()}
# for k, v in result_rm.items():
#   print(k + ": " + v)

#27
result_rm = {k : re.sub(r"\[\[(.*?)\]\]", r"\1", v) for k,v in result_rm.items()}
# for k, v in result_rm.items():
#   print(k + ": " + v)

#28
result_rm = {k : re.sub(r"<(.*?)>", "", v) for k,v in result_rm.items()}
# for k, v in result_rm.items():
#   print(k + ": " + v)

#29
import requests
n_flag = result_rm["国旗画像"]
filename = n_flag.replace(" ", "_")

URL = "https://www.mediawiki.org/w/api.php"

PARAMS = {
  "action": "query",
  "format": "json",
  "prop": "imageinfo",
  "titles": "File:" + filename,
  "iiprop": "url"
}

data = requests.get(url=URL, params=PARAMS)

print(re.search(r"\"url\"\:\"(.*?)\"", data.text).group())
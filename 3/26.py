import json
import re

def uk_extract():
    uk_text = []
    with open('jawiki-country.json\jawiki-country.json', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if article['title'] == "イギリス":
                uk_text.append(article['text'])
    return uk_text[0]

def template_extract(text):
    pat = r"\{\{基礎情報 国\n([\s|\S]*)\n\}\}" ###"([\s|\S]*)^\}\}"
    template = re.findall(pat, text)
    return template[0]
def remove_emphasis(text):
    return re.sub("'{2,}", "", text)
def main():
    uk_text = uk_extract()
    template = template_extract(uk_text)    
    pat = r"\|(.+?)\s*=\s*(.+?)(?:(?=\n$)|(?=\n\|)|(?=$))"
    results = re.findall(pat,template, re.MULTILINE+re.DOTALL)   
    results = {k:remove_emphasis(v) for k,v in results}
    for k,v in results.items():
        print(k,":",v)
if __name__ == "__main__":
    main()
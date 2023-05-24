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

def main():
    pat = r'(\={2,})\s*(.+?)\s*(\={2,})'
    uk_text = uk_extract()
    results = re.findall(pat,uk_text)
    for result in results:
        print(result[1].replace(" ", ""),":",len(result[0])-1)

if __name__ == "__main__":
    main()
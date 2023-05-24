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

def remove_link(text):
    pat = r"\[\[(?!ファイル)(.*)\]\]"
    if re.match(pat, text):
        # [[記事名]]
        text = re.sub(pat, r"\1", text)
        # [[記事名|表示名]]の場合
        pat = r"^.*?\|(.*?)"
        text = re.sub(pat, r"\1",text)
    return text

def remove_ref(text):
    pat = r"<ref name.*?>"
    text = re.sub(pat, "", text)
    pat = r"<ref>.*?</ref>"
    text = re.sub(pat, "", text)
    return text

def remove_stub(text):
    return re.sub(r"\{\{0\}\}", "", text)

def remove_url(text):
    return re.sub(r"\[http.*?]","",text)

def remove_file(text):
    return re.sub(r"\[\[ファイル:(.*?)\|.*?\]\]", r"\1", text)

def clean_text(text):
    text = remove_emphasis(text)
    text = remove_link(text)
    text = remove_ref(text)
    text = remove_stub(text)
    text = remove_url(text)
    text = remove_file(text)
    return text
def main():
    uk_text = uk_extract()
    template = template_extract(uk_text)    
    pat = r"\|(.+?)\s*=\s*(.+?)(?:(?=\n$)|(?=\n\|)|(?=$))"
    results = re.findall(pat,template, re.MULTILINE+re.DOTALL)   
    results = {k:clean_text(v) for k,v in results}
    for k,v in results.items():
        print(k,":",v)
if __name__ == "__main__":
    main()
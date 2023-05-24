import json
def uk_extract():
    uk_text = []
    with open('jawiki-country.json\jawiki-country.json', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if article['title'] == "イギリス":
                uk_text.append(article['text'])
    return uk_text[0]

def main():
    uk_text = uk_extract()
    print(uk_text)
if __name__ == "__main__":
    main()
import requests
def main():
    file_name = "Flag of the United Kingdom.svg"
    print(file_name)
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "File:"+file_name,
        "iiprop":"url"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA["query"]["pages"]
    print(PAGES['23473560']['imageinfo'][0]['url'])
if __name__ == "__main__":
    main()
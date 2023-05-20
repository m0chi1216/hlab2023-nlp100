def tango_bi_gram(st):
    st=st.replace(",","")
    st=st.replace(".","")
    stsp=st.split()
    gram =[]
    for i in range(len(stsp)-1):
        gram.append([stsp[i],stsp[i+1]])
    print(gram)

def moji_bi_gram(st):
    st=st.replace(",","")
    st=st.replace(".","")
    gram =[]
    for i in range(len(st)-1):
        gram.append([st[i],st[i+1]])
    print(gram)


def ngram(n, lst):
  # ex.
  # [str[i:] for i in range(2)] -> ['I am an NLPer', ' am an NLPer']
  # zip(*[str[i:] for i in range(2)]) -> zip('I am an NLPer', ' am an NLPer')
  #return list(zip(lst[0:],lst[1:]))
  return list(zip(*[lst[i:] for i in range(n)]))
#zipしたものはzip関数であるため、listでリスト型にする
#zipに二つ以上のリストを渡すとそのリストを前から順に統合していく
#今回zipに渡したリストは1つであるため、本来ならリストがzip化されるだけ
#そのzipにアスタリスクをつけることによって、

str = 'I am an NLPer'
words_bi_gram = ngram(2, str.split())
chars_bi_gram = ngram(2, str)

print('単語bi-gram:', words_bi_gram)
print('文字bi-gram:', chars_bi_gram)

st="I am an NLPer"
tango_bi_gram(st)
moji_bi_gram(st)




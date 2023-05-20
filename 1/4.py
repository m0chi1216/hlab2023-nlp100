st = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
st=st.replace(",","")
st=st.replace(".","")
st=st.lower()
stsp=st.split()

moji1=[1, 5, 6, 7, 8, 9, 15, 16, 19]
dict={}
count=0
for count,i in enumerate(stsp,1):
    if count in moji1:
        dict[i[0]]=count
    else:
        dict[i[:2]]=count

print(dict)
#enumerate関数はリストの内容と、その順番の番号をi,wordに入れる

str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
splits = str.split()
one_ch = [1, 5, 6, 7, 8, 9, 15, 16, 19]  # 1文字を取り出す単語の番号リスト
ans = {}
for i, word in enumerate(splits):
  if i + 1 in one_ch:
    ans[word[:1]] = i + 1  # リストにあれば1文字を取得
  else:
    ans[word[:2]] = i + 1  # なければ2文字を取得
ans
#スライス　進む　：　スライスする　進む　スライスする
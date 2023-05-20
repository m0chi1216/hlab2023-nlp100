st = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
st=st.replace(",","")
st=st.replace(".","")
stsp=st.split()
ans=[]
for i in stsp:
     ans.append(len(i))
ans2=[len(i)for i in stsp]
print(ans2)

import re

str = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
str = re.sub('[,\.]', '', str)  # ,と.を除去
splits = str.split()  # スペースで区切って単語ごとのリストを作成
ans = [len(i) for i in splits]
ans

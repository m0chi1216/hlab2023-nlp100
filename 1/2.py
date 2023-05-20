st1='パトカー'
st2='タクシー'
st = st1[0]+st2[0]+st1[1]+st2[1]+st1[2]+st2[2]+st1[3]+st2[3]
print(st)

str1 = 'パトカー'
str2 = 'タクシー'
ans = ''.join([i + j for i, j in zip(str1, str2)])  
# str1とstr2を同時にループ
# i+jしたものをjoin
print(ans)
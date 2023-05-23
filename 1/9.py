import random

st='I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
stsp = st.split()
shu_list=[]
for x in stsp:
    if len(x) >=4:
        ss=x[0]+''.join(random.sample(x[1:-1],len(x[1:-1])))+x[-1:]
        shu_list.append(ss)
    else:
        shu_list.append(x)

print(' '.join(shu_list))

def cipher(st):
    list1=[]
    '''
    for x in st:
        if x.islower:
            list1.append(chr(216-ord(x)))
        else:
            list1.append(x)
    return ''.join(list1)
    '''
    return ''.join([chr(219-ord(x)) if x.islower() else x for x in st])
    #リスト内包表記forでリストを作成、内容に関するif文はforの前に、リストに含むかどうかはforの後ろに
    rep = [chr(219-ord(x)) if x.islower() else x for x in st]
    
    return ''.join(rep)

message=cipher('the quick brown fox jumps over the lazy dog')
print('暗号化:', message)
message = cipher(message)
print('復号化:', message)
#意味的アナロジー，文法的アナロジーは検索してもヒットしない→一般的な用語？
with open('./questions-words-add.txt', 'r') as f:
  sem_cnt = 0
  sem_cor = 0
  #syn_cnt = 0
  #syn_cor = 0
  for line in f:
    line = line.split()
    sem_cnt += 1
    if line[4] == line[3]:
        sem_cor += 1

#question-words-addにカテゴリ名を含んでいなかったので計算できず 
#question-words-addは時間かかるので今回は飛ばす

print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
#print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}') 
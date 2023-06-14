from gensim.models import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt
import pprint

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


from tqdm import tqdm
with open('questions-words.txt', 'r') as f1:
    add_question = []
    for line in tqdm(f1, total=19558):
        word=line.split()
        if(word[0]!=':'):
            ans = model.most_similar(positive=[word[1], word[2]], negative=[word[0]],topn=1)[0]
            word.append(ans[0])
            word.append(str(ans[1]))
            add_question.append(word)

with open('questions-words-add.txt', 'w') as f1:
    for i in tqdm(range(len(add_question))):
        f1.write(' '.join(add_question[i]))
        f1.write('\n')
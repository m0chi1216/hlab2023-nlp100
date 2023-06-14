from gensim.models import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#query = 'United_States'
x= model['United_States']
y=model['U.S.']
cossim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
print(cossim)
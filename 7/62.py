from gensim.models import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print(model.most_similar('United_States', topn=10))
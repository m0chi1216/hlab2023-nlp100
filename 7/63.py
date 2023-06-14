from gensim.models import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt
import pprint

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
pprint.pprint(model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'],topn=10))
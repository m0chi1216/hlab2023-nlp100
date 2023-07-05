#question-words-add2は他の人からもらってきたやつ
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import bhtsne

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
cu = set()
with open('./questions-words-add2.txt', 'r') as f:
  for line in f:
    line = line.split()
    if line[0]=='currency':
      cu.add(line[1])
      cu.add(line[3])

countries = list(cu)
countries_vec=[model[i] for i in countries]

#bhtsne がインストールできなかったので試せてはない

embedded = bhtsne.tsne(np.array(countries_vec).astype(np.float64), dimensions=2, rand_seed=123)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, countries):
    plt.annotate(name, (x, y))
plt.show()
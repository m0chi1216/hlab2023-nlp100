#question-words-add2は他の人からもらってきたやつ
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

plt.figure(figsize=(10, 5))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
plt.show()
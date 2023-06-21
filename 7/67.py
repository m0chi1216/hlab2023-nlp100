#question-words-add2は他の人からもらってきたやつ
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

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

kmeans = KMeans(n_clusters=5) #, max_iter=30, init="random", n_jobs=-1
kmeans.fit(countries_vec)
clusters = kmeans.predict(countries_vec)

group = dict(zip(countries,clusters))

group0 = []
group1 = []
group2 = []
group3 = []
group4 = []

for i in countries:
  matchword = group[i]
  match matchword:
    case 0:
      group0.append(i)
    case 1:
      group1.append(i)
    case 2:
      group2.append(i)
    case 3:
      group3.append(i)
    case 4:
      group4.append(i)
    
print('group0')
print(group0)
print('group1')
print(group1)
print('group2')
print(group2)
print('group3')
print(group3)
print('group4')
print(group4)

from gensim.models import KeyedVectors

#60
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
#print(model["United_States"])

#61
simile = model.similarity("United_States", "U.S.")
print(simile)

#62
results = model.most_similar("United_States", topn=10)
print(results)
    
#63
spain = model["Spain"]
madrid = model["Madrid"]
athens = model["Athens"]
results=model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10)
print(results)

#64
# with open("questions-words.txt", "r") as f1, \
#     open("questions-words_add.txt", "w", encoding="utf-8_sig", newline="\n") as f2:
#     for line in f1:
#         line = line.split()
#         if line[0] == ":":
#             category = line[1]
#             f2.write(": " + category + "\n")
#         else:
#             word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
#             print(word,cos)
#             f2.write(" ".join([line[0], line[1], line[2], line[3], word, str(cos), "\n"]))


#65
with open("questions-words_add.txt", "r", encoding="utf-8_sig") as f:
    #意味的アナロジー：0, 文法的アナロジー:1
    semantic = []
    syntastic = []
    for line in f:
        line = line.split()
        if line[0] == ":":
            if "gram" in line[1]:
                category = 1
            else: category = 0            
        else:
            if category == 0:
                semantic.append(line[3:5])
            elif category == 1:
                syntastic.append(line[3:5])

    semantic_acc = sum([line[0] == line[1] for line in semantic])/len(semantic)
    syntastic_acc = sum([line[0] == line[1] for line in syntastic])/len(syntastic)
    print("semantic analogy accuracy: " + str(semantic_acc))
    print("syntastic analogy accuracy: " + str(syntastic_acc))


#66
from scipy.stats import spearmanr
import numpy as np
with open("wordsim353/combined.csv", "r", encoding="utf-8_sig") as f:
    next(f)
    human = []
    vector = []
    for line in f:
        line = line.rstrip("\n").split(",")
        human.append(float(line[2]))
        vector.append((model.similarity(line[0], line[1])))
    correlation, pvalue = spearmanr(human, vector)
    print("相関係数: ", correlation)
    print("p値: ", pvalue)


#67, 68, 69の準備
with open("questions-words.txt", "r") as f:
    countries = set()
    for line in f:
        line = line.split()
        if line[0] == ":":
            if line[1] == "currency":
                break
            else:
                continue
        else:
            countries.add(line[1])
    countries = list(countries)
    countries_vec = [model[country] for country in countries]

#67
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(countries_vec)
cluster = kmeans.labels_

for i in range(5):
    result = [country for j, country in enumerate(countries) if cluster[j] == i]
    print("category: ", i)
    for country in result:
        print(country, end=" ")
    print("\n")

#68
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
 
linkage_result = linkage(countries_vec, method='ward', metric='euclidean')
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=countries)
plt.show()

#69
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
countries_vec = np.array(countries_vec)
tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
X_tsne = tsne.fit_transform(countries_vec)

plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
plt.scatter(np.array(X_tsne).T[0], np.array(X_tsne).T[1])
for i in range(len(countries_vec)):
    plt.text(
        X_tsne[i, 0],
        X_tsne[i, 1],
        str(countries[i])
        )
plt.xlabel('t-SNE Feature1')
plt.ylabel('t-SNE Feature2')
plt.show()

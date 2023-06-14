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
with open("questions-words.txt", "r") as f1, \
    open("questions-words_add.txt", "w", encoding="utf-8_sig", newline="\n") as f2:
    for line in f1:
        line = line.split()
        if line[0] == ":":
            category = line[1]
            f2.write(category + "\n")
        else:
            word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            print(word,cos)
            f2.write(" ".join([line[0], line[1], line[2], line[3], word, str(cos), "\n"]))
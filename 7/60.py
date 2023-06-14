from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print(model['United_States'])
'''
new_vec = model['king'] - model['man'] + model['woman']
print(model.similar_by_vector(new_vec))
print(model.similar_by_vector(model['king']))
'''
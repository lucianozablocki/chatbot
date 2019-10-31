from gensim.models.keyedvectors import KeyedVectors
wordvectors_file_vec = 'fasttext-sbwc.3.6.e20.vec'
cantidad = 100000
wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)

#most_similar_cosmul -> me da las palabras mas cercanas a positive y alejadas de negative
print(wordvectors.most_similar_cosmul(positive=['rey','mujer'],negative=['hombre'])[0])
print(wordvectors.most_similar_cosmul(positive=['ir','jugando'],negative=['jugar'])[0])

#Me da la palabra que menor relación tiene con los otros
print(wordvectors.doesnt_match(['blanco','azul','rojo','perro']))

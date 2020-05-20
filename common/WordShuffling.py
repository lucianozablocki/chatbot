from nltk import word_tokenize
import random
import pandas as pn

def augment(sentence,n): #Recibo las oraciones, y la cantidad de veces que las mezclo
    new_sentences = []
    words = word_tokenize(sentence)
    for i in range(n):
        random.shuffle(words)
        new_sentences.append(' '.join(words))
    new_sentences = list(set(new_sentences))
    return new_sentences

preguntas = pn.read_csv("preguntas.csv",header=None)
preg = preguntas.values

# print(type(preg))

text = []
for i in range(preg.shape[0]):
    aux = preg[i][1].replace('"', '') #Borro las comillas
    text.append(aux)

nsentences = []

for i in range(2):
    nsentences[i] = augment(text[i],3)
for s in nsentences:
    print (s)
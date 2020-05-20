from nltk import word_tokenize
from random import randrange
from nltk.corpus import stopwords
import pandas as pn

stoplist = stopwords.words('spanish')
# print(stoplist)
def synonym_replacement(sentence, synonyms_lexicon):

    words = word_tokenize(sentence)
    # print("Words: ", words)
    n_sentence = sentence
    for w in words:
        cont = 0
        for k in synonyms_lexicon:
            if w not in stoplist:
                if w in k:
                    num = randrange(len(k))
                    # print("Quiero reemplazar: ",w)
                    while not synonyms_lexicon[cont][num] != w:
                        num = randrange(len(k))
                    n_sentence = n_sentence.replace(w, synonyms_lexicon[cont][num])  # we replace with the first synonym
                    # print("Reemplazo con: ",synonyms_lexicon[cont][num])
                cont+=1
    return n_sentence

def get_synonyms_lexicon(path):
    synonyms_lexicon = []
    text_entries = [l.strip() for l in open(path, encoding="utf8").readlines()]
    for e in text_entries:
        e = e.split(' ')
        e = [s.replace(',', '') for s in e]
        # print("Esto es e: ",e)
        synonyms_lexicon.append(e)
    return synonyms_lexicon

path = 'synonyms_es.txt'
# path = 'test.txt'
syns = get_synonyms_lexicon(path)
# print(syns)

preguntas = pn.read_csv("preguntas.csv",header=None)
preg = preguntas.values

# print(type(preg))

text = []
for i in range(preg.shape[0]):
    aux = preg[i][1].replace('"', '') #Borro las comillas
    text.append(aux)

for i in range(2):
    print("Frase actual: ",text[i])
    nuevaFrase = synonym_replacement(text[i],syns)
    print("Nueva frase: ",nuevaFrase)


from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pn
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker


stoplist = stopwords.words('spanish')
spanishStem=SnowballStemmer('spanish')
spell = SpellChecker(language='es')

# print(spanishStem.stem('legalmente'))



# preguntas = pn.read_csv("pregTest.csv",header=None)
# preg = preguntas.values
# print(preg)

def limpiarSignos(preguntas):
    text = []
    for i in range(preguntas.shape[0]):

        aux = preguntas[i][1].replace('"', '') #Borro las comillas
        aux = aux.replace('?','') #Borro signos de preguntas
        aux = aux.replace('¿','')
        aux = aux.replace(',','') #Borro comas
        aux = aux.replace('á','a') #Reemplazo signos de puntuación...
        aux = aux.replace('é','e')
        aux = aux.replace('í','i')
        aux = aux.replace('ó','o')
        aux = aux.replace('ú','u')
        preguntas[i][1] = aux
    return (preguntas)

def quitarStopwords(preguntas):
    for i in range(preguntas.shape[0]): #Para todas las oraciones...
        words = word_tokenize(preguntas[i][1]) #Separo en palabras
        oracion = '' #Vacio el vector de palabras de la oracion actual
        for w in words: 
            if w not in stoplist: #Filtro las stopwords...
                oracion = oracion + w + ' '

        preguntas[i][1] = oracion #Dejo en la matriz la oracion limpia
    return (preguntas)

def Stemmizar (preguntas):
   for i in range(preguntas.shape[0]):
       words = word_tokenize(preguntas[i][1])
       oracion = '' #Vacio el vector de palabras de la oracion actual
       for w in words: 
           palabraStem = spanishStem.stem(w)
           oracion = oracion + palabraStem + ' '
       preguntas[i][1] = oracion
   return(preguntas)

def Autocorrector (preguntas): #Recibe una matriz, es para corregir la matriz de preguntas
    for i in range(preguntas.shape[0]):
        words = word_tokenize(preguntas[i][1])
        # print(words)
        oracion = ''

        for w in words:
            mispelled = spell.unknown(words)
            # print("LO INCORRECTO ES: ",mispelled)
            for word in mispelled:
                # print("Quiero corregir: ",word)
                corregida = spell.correction(word)
                preguntas[i][1] = preguntas[i][1].replace(word,corregida)

    return (preguntas)

def AutocorrectorInput (sentence): #Corrije de a una lista de palabras (como para usar en la entrada del usuario)
    #Me fijo las palabras que pueden llegar a estar mal escritas....
    mispelled = spell.unknown(sentence)
    # print(mispelled)
    for word in mispelled: #Corrijo cada una, y la reemplazo en la lista
        corregida = spell.correction(word)
        # print("Corregi algo:",corregida)
        sentence = [corregida if x==word else x for x in sentence]
        # print(sentence)
 
        # sentence = sentence.replace(word,corregida)

    return(sentence)

text = []
def preprocesar(preguntas):

    preguntas = limpiarSignos(preguntas)
    preguntas = quitarStopwords(preguntas) #NOTAR! Los brackets quedan como:    < carrera >
    preguntas = Autocorrector(preguntas)
    # print("LO CORREGIDO ES: ",preguntas)
    preguntas = Stemmizar(preguntas)

    # print(preguntas)


# preprocesar(preg)
sentenceTest  = ['holax','coomo','estass']
print(sentenceTest)
print(AutocorrectorInput(sentenceTest))
# nltk.download()

# nltk.stem.snowball.demo()
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pn
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker
import spacy

nlp = spacy.load('es_core_news_sm')
stoplist = stopwords.words('spanish')
spanishStem=SnowballStemmer('spanish')
spell = SpellChecker(language='es')

def Lematizar(preguntas): #Recibo matriz de preguntas/respuestas
    for i in range(preguntas.shape[0]):
        oracion = ''
        for token in nlp(preguntas[i][1]):
            oracion = oracion + token.lemma_ + ' '
            # print(token.text, token.lemma_, token.pos_)
        preguntas[i][1] = oracion
    return (preguntas)

def LematizarOracion(sentence): #Recibo string
    oracion = ''
    for token in nlp(sentence):
        oracion = oracion + token.lemma_ + ' '
    return (oracion)


def limpiarSignos(preguntas):
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

#metodo que elimina las stopwords de un comentario
def remove_stopwords(preguntas, stopwords):
    n=preguntas.shape[0]
    resultado=[]
    for i in range(n):
        sentence = preguntas[i][1]        
        sentencewords = sentence.split() #divide el comentario en una lista de palabras
        resultwords  = [word for word in sentencewords if word.lower() not in stopwords] #de las palabras que haya en sentenceword, devolveme las que no esten en stopwords
        result = ' '.join(resultwords) #une la lista que se genero antes (sin las stopwords)
        resultado.append(result)
        resultado.append("\n")
    return resultado

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
        sentence = [corregida if x==word else x for x in sentence]
    return(sentence)

def preprocesar(preguntas):

    preguntas = Lematizar(preguntas)
    preguntas = limpiarSignos(preguntas)
    #preguntas_otras = remove_stopwords(preguntas,stoplist)
    preguntas = quitarStopwords(preguntas) #(!) IMPORTANTE! Los brackets quedan como: < carrera >
    preguntas = Autocorrector(preguntas)
    # preguntas = Stemmizar(preguntas) <--- Descomentar esto, y comentar el Lematizador
    return (preguntas)


preguntas = pn.read_csv("pregTest.csv",header=None)
preg = preguntas.values
print(preg)
preprocesadas = preprocesar(preg)
print(preprocesadas)

#lematizadas = Lematizar(preg)
#print(lematizadas)
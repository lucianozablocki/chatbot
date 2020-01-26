from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pn
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker
import spacy
import time

nlp = spacy.load('es_core_news_sm')
stoplist = stopwords.words('spanish')
spanishStem=SnowballStemmer('spanish')
spell = SpellChecker(language='es')

def Lematizar(preguntas): #Recibo matriz de preguntas/respuestas
    t = time.time()
    for i in range(preguntas.shape[0]):
        oracion = ''
        for token in nlp(preguntas[i][1]):
            oracion = oracion + token.lemma_ + ' '
            # print(token.text, token.lemma_, token.pos_)
        preguntas[i][1] = oracion
    print('Elapsed in lematizar: ' ,(time.time() - t))
    return (preguntas)

def LematizarOracion(sentence): #Recibo string
    oracion = ''
    for token in nlp(sentence):
        oracion = oracion + token.lemma_ + ' '
    return (oracion)

def limpiarSignosinput(inp):
    aux = inp.replace('"', '')
    aux = aux.replace('?','') #Borro signos de preguntas...
    aux = aux.replace('¿','')
    aux = aux.replace('!','')#.. y signos de exclamacion
    aux = aux.replace('¡','')
    aux = aux.replace(',','') #Borro comas
    aux = aux.replace('á','a') #Reemplazo signos de puntuación...
    aux = aux.replace('é','e')
    aux = aux.replace('í','i')
    aux = aux.replace('ó','o')
    aux = aux.replace('ú','u')
    aux = aux.lower() #Llevo todo a minuscula
    return aux
def limpiarSignos(preguntas):
    t = time.time()
    for i in range(preguntas.shape[0]):

        aux = preguntas[i][1].replace('"', '') #Borro las comillas
        aux = aux.replace('?','') #Borro signos de preguntas...
        aux = aux.replace('¿','')
        aux = aux.replace('!','')#.. y signos de exclamacion
        aux = aux.replace('¡','')
        aux = aux.replace(',','') #Borro comas
        aux = aux.replace('á','a') #Reemplazo signos de puntuación...
        aux = aux.replace('é','e')
        aux = aux.replace('í','i')
        aux = aux.replace('ó','o')
        aux = aux.replace('ú','u')
        aux = aux.lower() #Llevo todo a minuscula
        preguntas[i][1] = aux
    print('Elapsed in limpiarsignos: ' ,(time.time() - t))
    return (preguntas)

def quitarStopwords(preguntas):
    t = time.time()
    for i in range(preguntas.shape[0]): #Para todas las oraciones...
        words = word_tokenize(preguntas[i][1]) #Separo en palabras
        oracion = '' #Vacio el vector de palabras de la oracion actual
        for w in words: 
            if w not in stoplist: #Filtro las stopwords...
                oracion = oracion + w + ' '

        preguntas[i][1] = oracion #Dejo en la matriz la oracion limpia
    print('Elapsed in quitarstopwords: ' ,(time.time() - t))    
    return (preguntas)
def quitarStopwordsinput(input):
    ret = ''
    for w in input:
        if w not in stoplist:
            print(w)
            ret = ret + w + ''
        else:
            print("erasing word")
    return ret
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
        #resultado.append("\n")
    return resultado
def Stemmizarinput(inp):
    ret = ''
    for w in inp:
        palabraStem = spanishStem.stem(w)
        ret = ret + palabraStem + ''
    return ret
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
    t =time.time()
    for i in range(preguntas.shape[0]):
        print(i)
        words = word_tokenize(preguntas[i][1])
        # print(words)
        
        mispelled = spell.unknown(words)
        # print("LO INCORRECTO ES: ",mispelled)
        for word in mispelled:
            # print("Quiero corregir: ",word)
            corregida = spell.correction(word)
            preguntas[i][1] = preguntas[i][1].replace(word,corregida)
    print('Elapsed in autocorrector: ' ,(time.time() - t))
    return (preguntas)

def AutocorrectorInput (sentence): #Corrije de a una lista de palabras (como para usar en la entrada del usuario)
    #Me fijo las palabras que pueden llegar a estar mal escritas....
    mispelled = spell.unknown(sentence)
    # print(mispelled)
    for word in mispelled: #Corrijo cada una, y la reemplazo en la lista
        corregida = spell.correction(word)
        sentence = [corregida if x==word else x for x in sentence]
    return(sentence)

def preprocesar(preguntas,tipo): 

    preguntas = quitarStopwords(preguntas) #(!) IMPORTANTE! Los brackets quedan como: < carrera >
    if tipo==1:
        print("---init lemmatization---")
        preguntas = Lematizar(preguntas)
        print("---finished lemmatization--")
    print("---init limpiarSignos---")
    preguntas = limpiarSignos(preguntas)
    print("---finished limpiarSignos---")
    #preguntas_otras = remove_stopwords(preguntas,stoplist)
    #print("---init quitarStopwords---")
    #print("---finished quitarStopwords---")
    print("---init autocorrector---")
    preguntas = Autocorrector(preguntas)
    # print("hola")
    print("---finished autocorrector---")
    if tipo==2:
        print("---init stemming---")
        preguntas = Stemmizar(preguntas) #<--- Descomentar esto, y comentar el Lematizador
        print("---finished stemming---")
    
    return (preguntas)

if __name__ == "__main__": #modificar metodos para devolver preguntas preprocesadas en formato esperado: lista
                            #de preguntas separadas por coma -> ["como estudio","vivo en extranjero","como dar de baja"]
    dataset = pn.read_csv("preguntas.csv",header=None,delimiter=',')
    labels = dataset.values[:,0]
    cantidadLabels = dataset.values[len(dataset.values)-1,0] + 1
    print("Terminé de cargar los datos recién...")
    # print(labels)
    ################
    ################
    print("Arranco a preprocesar...")
    correctedData = preprocesar(dataset.values,1) #Dataset lematizado
    #print(correctedData)
    ################
    ################
    print("Guardando como csv...")
    df_correctedData = pn.DataFrame(correctedData)
    df_correctedData.to_csv('preprocessedQuestions_lem.csv', index=False,header=False)

    dataset = pn.read_csv("preprocessedQuestions_lem.csv",header=None,delimiter=',')
    print(dataset.shape)
    
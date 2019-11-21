import numpy as np
from nltk import word_tokenize
import torch 


def contarClase(dataset,clase): #metodo que cuenta la cantidad de patrones en 'dataset' que pertenecen a la clase 'clase'
    contador = 0
    for i in range(dataset.shape[0]):
        if dataset[i,0]==clase:
            contador+=1
    return contador

def create_tensor_prom_and_embedding(Xtext,cant_patrones,wordvectors):
    token_idx = 0
    wordSet = {} #diccionario de tokens. clave -> token valor -> indice de la posicion en el tensor de embeddings
    embeddingMat = [] #array (que luego sera convertido a tensor) de embeddings vocabulario propio
    promMat = np.zeros((300,cant_patrones),dtype=np.float32) #array de promedios de vectores de embeddings vocabulario original -> decision: no promediar embeddings no conocidos
    for i in range(cant_patrones):
        words = word_tokenize(Xtext[i,0])
        words_in_sentence = [] #lista de indices de las palabras que se encuentran en la oracion actual
        prom = np.zeros((300)) #acumulador de vectores de embeddings de las palabras de la oracion actual
        for w in words: #w is a token
            if w not in wordSet: #la palabra no se encuentra en el diccionario generado: se debe agregar
                wordSet[w] = token_idx
                if w in wordvectors:
                    words_in_sentence.append(token_idx)
                    s = wordvectors.get_vector(w)
                    embeddingMat.append(s) #Acá hay que meter al tensor en verdad. -> precision en los decimales
                    token_idx += 1
                else: #la palabra no esta en el vocabulario original, agregamos un vector de numeros aleatorios que luego sera entrenado
                    embeddingMat.append(np.random.normal(0,0.2,300))
                    token_idx+=1
            else: #la palabra ya se encuentra en el diccionario generado: debemos asignarle a la lista de indices el indice del token corresp
                words_in_sentence.append(wordSet[w])
        for indx in words_in_sentence: #promediamos solo los vectores de embedding encontrados en el vocabulario original
            prom += embeddingMat[indx] #Calcular promedio con embeddingMat y los indices de words_in_sentence
        cant_palabras = len(words_in_sentence)
        if len(words_in_sentence)>0:
            promMat[:,i] = prom/cant_palabras

    tensorEmbedding = torch.Tensor(embeddingMat)
    promMat = np.transpose(promMat)
    X = torch.from_numpy(promMat)
    X = X.float()
    return X,tensorEmbedding,wordSet

def separate_dataset(correctedData,cantidad_preg):
    #Separo el dataset en test y train -> un patron de cada clase al subconjunto de test, el resto a train
    clase_actual = -1
    indxTest = [] #lista de indices de preguntas en subconjunto de test
    indxTrain= [] #lista de indices de preguntas en subconjunto de train
    for i in range(cantidad_preg):
        if correctedData[i,0]!=clase_actual: #cambio de clase
            clase_actual = correctedData[i,0]
            cantidadPregClase = contarClase(correctedData,clase_actual) #cuento cuantas preguntas hay pertenecientes a la clase actual
            if clase_actual!=46 and clase_actual !=103 and clase_actual!=104 and clase_actual !=105: #clases con solo 1 patron no se incluyen en test
                indxTest.append(np.random.randint(0, cantidadPregClase-1)+i) #tomo un indice al azar por clase y lo agrego al subconjunto de test
    #Obtengo los indices de train
    indices = range(cantidad_preg)
    indxTrain = list(filter(lambda x: x not in indxTest, indices)) #al no meter clases con un solo patron al indxTest, pasan automaticamente al indxTrain
    return indxTest,indxTrain

from gensim.models.keyedvectors import KeyedVectors
import gensim
import pandas as pn
from nltk import word_tokenize
import numpy as np
# import sys
# sys.path.append('C:/Users/Juani/chatbot/')
import preprocesamiento

class embeddings():
    def __init__(self,wordvectors,correctedData,tipoEmbed):
        self.tipoEmbed = tipoEmbed

    def train(self):
        vec_embeddings=[]
        for i in range(correctedData.shape[0]):
            words = word_tokenize(correctedData[i][1])
            # print(words)
            vec_sentence = []
            for w in words:
                s = wordvectors.get_vector(w)
                if type(s)!=int:
                    # print("Encontró ",w)
                    vec_sentence.append(s)
            prom = np.mean(vec_sentence, axis=0)
            if self.tipoEmbed==1:
                vec_embeddings.append(vec_sentence)
            elif self.tipoEmbed ==2:
                vec_embeddings.append(prom)
            else:
                vec_sentence.append(prom)
                vec_embeddings.append(vec_sentence)
        return(vec_embeddings)


if __name__=='__main__':
    print("-----------------Cargando los vectores----------------")
    wordvectors_file_vec = 'C:/Users/Juani/chatbot/fasttext-sbwc.3.6.e20.vec'
    cantidad = 100000
    wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
    print("Listo!")
    print("-----------------Cargando las preguntas---------------")
    dataset = pn.read_csv("pregTest.csv",header=None,delimiter=',')
    correctedData = preprocesamiento.preprocesar(dataset.values,1) #Dataset lematizado
    print("Listo. Las preguntas son:  ")
    print(correctedData)
    print("------------------Creando embeddings------------------")
    embed = embeddings(wordvectors,correctedData,1)
    embeddedWords = embed.train()
    print("Listo! Embeddings creados correctamente! Por ejemplo, el primero es:")
    print(embeddedWords[0])
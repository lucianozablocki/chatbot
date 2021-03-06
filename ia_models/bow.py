from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
import pandas
#import preprocesamiento #los metodos de preprocesamiento.py reciben datos en la forma: data.values donde data
                        #es producto de leer csv con pandas.read_csv
# import torch as tr

class BOW():
    def __init__(self,dataset,strip_accents,stoplist,weighting,ngram = None):
        #autocorregir -> lematizar -> borrar signos, carac especiales, stopwords, pasar a minuscula
        #x_text_auto = preprocesamiento.Autocorrector(dataset.values)
        #x_text_lem = preprocesamiento.Lematizar(dataset)
        #x_text_lem = x_text_lem[:,1]
        #x_text_lem = dataset[:,1] esto anda
        if weighting:
            if ngram: 
                vectorizer = TfidfVectorizer(strip_accents=strip_accents,stop_words=stoplist,ngram_range=ngram) 
            else:
                vectorizer = TfidfVectorizer(strip_accents=strip_accents,stop_words=stoplist)   
        else:
            if ngram:
                vectorizer = CountVectorizer(strip_accents=strip_accents,stop_words=stoplist,ngram_range=ngram)     
            else: 
                vectorizer = CountVectorizer(strip_accents=strip_accents,stop_words=stoplist)
        #X = vectorizer.fit_transform(x_text_lem)  
        X = vectorizer.fit_transform(dataset)
        self.vectorizer = vectorizer
        self.X = X

    def get_vocab(self):
        return(self.vectorizer.get_feature_names())

if __name__ == '__main__':
    #algo = tr.tensor([[-1, -1], [2, 1]])
    #print(algo)
    
    stoplist = stopwords.words('spanish')
    dataset = pandas.read_csv("pregTest.csv",header=None)
    #n_gram = (1,2)
    y = dataset.values[:,0]
    print(y)
    #bow_train_bigram = BOW(dataset,'ascii',stoplist,False,ngram=(2,2))
    #print(bow_train_bigram.get_vocab())
    #print(bow_train.X.shape)
    #print(type(bow_train.X))
    bow_train_unigram = BOW(dataset,'ascii',stoplist,False)
    print(bow_train_unigram.get_vocab())
    print(bow_train_unigram.X.shape)
    
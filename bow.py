from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
import pandas
import preprocesamiento #los metodos de preprocesamiento.py reciben datos en la forma: data.values donde data
                        #es producto de leer csv con pandas.read_csv

class BOW():
    def __init__(self,dataset,strip_accents,stoplist,weighting,ngram = None):
        #autocorregir -> lematizar -> borrar signos, carac especiales, stopwords, pasar a minuscula
        x_text_auto = preprocesamiento.Autocorrector(dataset.values)
        x_text_lem = preprocesamiento.Lematizar(x_text_auto)
        x_text_lem = x_text_lem[:,1]
        if weighting:
            if ngram: 
                vectorizer = TfidfVectorizer(strip_accents=strip_accents,stop_words=stoplist,ngram_range=ngram) 
            else:
                vectorizer = TfidfVectorizer(strip_accents=strip_accents,stop_words=stoplist)   
            X = vectorizer.fit_transform(x_text_lem)
        else:
            if ngram:
                vectorizer = CountVectorizer(strip_accents=strip_accents,stop_words=stoplist,ngram_range=ngram)     
            else: 
                vectorizer = CountVectorizer(strip_accents=strip_accents,stop_words=stoplist)
            X = vectorizer.fit_transform(x_text_lem)  

        self.vectorizer = vectorizer
        self.X = X

if __name__=='__main__':
    stoplist = stopwords.words('spanish')
    dataset = pandas.read_csv("pregTest.csv",header=None,delimiter=',')
    #n_gram = (1,2)
    y = dataset.values[:,0]
    print(y)
    bow_train = BOW(dataset,'ascii',stoplist,False)
    print(bow_train.vectorizer.get_feature_names())
    print(bow_train.X.shape)
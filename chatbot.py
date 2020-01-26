import sys, pandas as pd, pickle, numpy as np, preprocesamiento,bow
from preprocesamiento import quitarStopwordsinput,limpiarSignosinput,AutocorrectorInput,Stemmizarinput
from nltk.corpus import stopwords
filename = 'C:/Users/lucy/chatbot/SVC_entrenado.sav'
path_ans = 'C:/Users/lucy/chatbot/respuestas - respuestas.csv'
path_adjMat = 'C:/Users/lucy/chatbot/adjMat.csv'
path_quest = 'C:/Users/lucy/chatbot/preprocessedQuestions_stem_completadas.csv'

#print(len(sys.argv))
#print(sys.argv)
R_pd = pd.read_csv(path_ans,delimiter=',',header=None)
adjMat_pd = pd.read_csv(path_adjMat,delimiter=',',header=None)
R = R_pd.values
adjMat = adjMat_pd.values
#print(adjMat.shape)
actual_node = 109
loaded_model = pickle.load(open(filename, 'rb'))
#loaded_bow = pickle.load(open(path_bow,'rb'))
thres = 0.5

stoplist = stopwords.words('spanish')
dataset = pd.read_csv(path_quest,header=None)
bow_unigram = bow.BOW(dataset.values[:,1],'ascii',stoplist,True)

def preprocesar(input):
    i = quitarStopwordsinput([input])
    i = limpiarSignosinput(i)
    i = AutocorrectorInput([i])
    i = Stemmizarinput([i])
    return i

while(True):
    print(R[actual_node][1])
    i = input()
    pre_input = preprocesar(i)
    print("after preproc: ", pre_input)
    model_input = bow_unigram.vectorizer.transform([pre_input])
    probs = loaded_model.predict_proba(model_input) #Esto es de mentirita, estoy prediciendo contra datos de train (chanchan)
    #print(probs)
    if(any(x > thres for x in probs[0])):
        probs_flux = probs*adjMat[actual_node][0:105]
        next_node = np.argmax(probs_flux)
        actual_node = next_node
    elif R[actual_node][2] == 1: #administrative error
        actual_node = 107
    elif R[actual_node][2] == 2: #academic error
        actual_node = 108
    else: #general error
        actual_node = 106

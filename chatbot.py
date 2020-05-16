import sys, pandas as pd, pickle, numpy as np, preprocesamiento,bow
from preprocesamiento import quitarStopwordsinput,limpiarSignosinput,AutocorrectorInput,Stemmizarinput
from nltk.corpus import stopwords
filename = 'C:/Users/lucy/chatbot/SVC_entrenado.sav'
path_ans = 'C:/Users/lucy/chatbot/respuestas - respuestas.csv'
path_adjMat = 'C:/Users/lucy/chatbot/adjMat.csv'
path_quest = 'C:/Users/lucy/chatbot/preprocessedQuestions_stem_completadas.csv'
path_placeholders =  'C:/Users/lucy/chatbot/placeholders.csv'

#print(len(sys.argv))
#print(sys.argv)
R_pd = pd.read_csv(path_ans,delimiter=',',header=None)
adjMat_pd = pd.read_csv(path_adjMat,delimiter=',',header=None)
R = R_pd.values
adjMat = adjMat_pd.values
placeholders_pd = pd.read_csv(path_placeholders,delimiter=',',header=None)
placeholders = placeholders_pd.values
# print(adjMat.shape)
actual_node = 109
loaded_model = pickle.load(open(filename, 'rb'))
#loaded_bow = pickle.load(open(path_bow,'rb'))
thres = 0.1

# nltk.download('stopwords')
stoplist = stopwords.words('spanish')
dataset = pd.read_csv(path_quest,header=None)
bow_unigram = bow.BOW(dataset.values[:,1],'ascii',stoplist,True)

keys = placeholders[:,0]
values = placeholders[:,1]
replacements = dict(zip(keys,values))
# print(replacements)

def preprocesar(sentence):
    # print(f"preprocesar method, input its: {sentence}")
    i = quitarStopwordsinput(sentence.split()) #we are using whitespaces as separator, potential problem
    # print(f"output after quitarstopwords and input to limpiarsignos: {i}")
    #num parameter in split can delimite the number of elements in the resulting list
    i = limpiarSignosinput(i)
    # print(f"output after limpiarsignos and input to autocorrector: {i.split()}")
    i = AutocorrectorInput(i.split())
    # print(f"output after autocorrector and input to stemming: {i}")
    i = Stemmizarinput(i)
    return i.strip()

while(True):
    print(f"actual node is: {actual_node}")
    for placeholder in replacements:
        R[actual_node][1] = R[actual_node][1].replace(f'<{placeholder}>', replacements[placeholder])
    print(R[actual_node][1])
    i = input()
    pre_input = preprocesar(i)
    print("input after preproc: ", pre_input)
    model_input = bow_unigram.vectorizer.transform([pre_input])
    probs = loaded_model.predict_proba(model_input)
    print(f"max prob is {np.max(probs)}")
    # print(probs.shape)
    # print(np.argmax(probs))
    # print(probs)
    if(any(x > thres for x in probs[0])):
        if actual_node == 109:
            probs_flux = probs
        else:
            probs_flux = probs*adjMat[actual_node][0:106]
        next_node = np.argmax(probs_flux)
        actual_node = next_node
    elif R[actual_node][2] == 1: #administrative error
        actual_node = 107
    elif R[actual_node][2] == 2: #academic error
        actual_node = 108
    else: #general error
        actual_node = 106
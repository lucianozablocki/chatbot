import sys, pandas as pd, pickle, numpy as np, preprocesamiento,bow
from preprocesamiento import quitarStopwordsinput,limpiarSignosinput,AutocorrectorInput,Stemmizarinput
from nltk.corpus import stopwords
import sklearn
filename = './SVC_stem.sav'
path_ans = './respuestas - respuestas.csv'
path_adjMat = './new_adjMat.csv'
# path_quest = './preprocessedQuestions_stem_completadas.csv'
path_placeholders =  './placeholders.csv'
path_bow = "./bow_w_stopwords.sav"

print(f'The scikit-learn version is {sklearn.__version__}.')

R_pd = pd.read_csv(path_ans,delimiter=',',header=None)
adjMat_pd = pd.read_csv(path_adjMat,delimiter=',',header=None)
R = R_pd.values
adjMat = adjMat_pd.values
placeholders_pd = pd.read_csv(path_placeholders,delimiter=',',header=None)
placeholders = placeholders_pd.values
# print(adjMat.shape)
actual_node = 115
loaded_model = pickle.load(open(filename, 'rb'))
bow_unigram = pickle.load(open(path_bow, 'rb'))
thres = 0.01

stoplist = stopwords.words('spanish')
keys = placeholders[:,0]
values = placeholders[:,1]
replacements = dict(zip(keys,values))

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

def obtenerNodo(case):
    switcher = {
        "bachiller" : 69,
        "licenciatura" : 106,
        "curso": 107,
        "tecnicatura": 108,
        "posgrado": 109,
        "reconquista": 88,
        "galvez": 110,
        "rafaela": 111
    }

    return switcher.get(case, 107) #default->error adm

while(True):
    print(f"actual node is: {actual_node}")
    for placeholder in replacements:
        R[actual_node][1] = R[actual_node][1].replace(f'<{placeholder}>', replacements[placeholder])
    print(R[actual_node][1])
    i = input()
    pre_input = preprocesar(i)
    print("input after preproc: ", pre_input)
    model_input = bow_unigram.transform([pre_input])
    probs = loaded_model.predict_proba(model_input)
    print(f"max prob is {np.max(probs)}")
    print(f"predicted node is {np.argmax(probs)}")
    
    if(any(x > thres for x in probs[0])):
        if actual_node == 115:
            probs_flux = probs
        else:
            probs_flux = probs*adjMat[actual_node][0:106]
        next_node = np.argmax(probs_flux)
        actual_node = next_node
        if actual_node == 69:
            print("Por favor, decime que carrera queres estudiar")
            carrera = input()
            actual_node = obtenerNodo(carrera)
        elif actual_node == 88:
            print("Por favor, decime con que sede te queres contactar")
            sede = input()
            actual_node = obtenerNodo(sede)
        elif actual_node == 97 or actual_node == 37:
            print("Por favor, ingresa el contexto al cual pertenece tu pregunta (academico o administrativo)")
            contexto = input()
            actual_node = 37 if contexto == "administrativo" else 97
    elif R[actual_node][2] == 1: #administrative error
        actual_node = 113
    elif R[actual_node][2] == 2: #academic error
        actual_node = 114
    else: #general error
        actual_node = 112
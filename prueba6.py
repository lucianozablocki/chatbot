from gensim.models.keyedvectors import KeyedVectors
import sys
sys.path.append('C:/Users/Juani/chatbot/')
#import gensim
#import pandas as pn
from nltk import word_tokenize
import numpy as np
#import preprocesamiento
import torch
print(torch.__version__)
#import torchvision.transforms as transforms
#from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
#from torchvision import datasets
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
#import utils
#from utils import EarlyStopping
from sklearn.metrics import balanced_accuracy_score
from config import indxTrain,indxTest,indxVal
import skorch
from skorch import NeuralNetClassifier,NeuralNet
from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
#/////////////////////////////////////////////////////////////

def generarDiccionario(dataset,wordvectors): #Recibo el dataset.values y los vectores
    
    token_idx = 0
    cant_patrones = dataset.shape[0]
    dataset = adaptarDataset(dataset) #esto pasa de vectores tipo ['holi' 'perri'] a [['holi'] ['perri']] que es lo que necesito
    wordSet = {} #diccionario de tokens. clave -> token valor -> indice de la posicion en el tensor de embeddings
    embeddingMat = [] #array (que luego sera convertido a tensor) de embeddings vocabulario propio
    promMat = np.zeros((300,cant_patrones),dtype=np.float32) #array de promedios de vectores de embeddings vocabulario original -> decision: no promediar embeddings no conocidos
    for i in range(cant_patrones):
        #print(dataset[i,0])
        words = word_tokenize(dataset[i,0])
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

#/////////////////////////////////////////////////////////////

def adaptarDataset(dataset):
    cant_patrones = dataset.shape[0]
    aux= np.zeros((cant_patrones,1),dtype=object)
    for i in range(cant_patrones):
        aux[i] = dataset[i,1]
    return aux

#/////////////////////////////////////////////////////////////

def contarClase(dataset,clase): #metodo que cuenta la cantidad de patrones en 'dataset' que pertenecen a la clase 'clase'
    contador = 0
    for i in range(dataset.shape[0]):
        if dataset[i,0]==clase:
            contador+=1
    return contador

#/////////////////////////////////////////////////////////////

def separate_dataset(correctedData,validation=True):
    #Separo el dataset en test y train -> un patron de cada clase al subconjunto de test, el resto a train
    cantidad_preg = correctedData.shape[0]
    clase_actual = -1
    indxTest = [] #lista de indices de preguntas en subconjunto de test
    indxTrain= [] #lista de indices de preguntas en subconjunto de train
    indxVal = []
    #Xval = None
    #Yval = None
    
    for i in range(cantidad_preg):
        if correctedData[i,0]!=clase_actual: #cambio de clase
            clase_actual = correctedData[i,0]
            cantidadPregClase = contarClase(correctedData,clase_actual) #cuento cuantas preguntas hay pertenecientes a la clase actual
            if clase_actual!=46 and clase_actual !=103 and clase_actual!=104 and clase_actual !=105: #clases con solo 1 patron no se incluyen en test
                indxTest.append(np.random.randint(0, cantidadPregClase-1)+i) #tomo un indice al azar por clase y lo agrego al subconjunto de test
                if validation:
                    insertIndx = True
                    while insertIndx:
                        j = np.random.randint(0, cantidadPregClase-1)+i
                        if j not in indxTest:
                            indxVal.append(j)
                            insertIndx = False

    #Obtengo los indices de train
    indices = range(cantidad_preg)
    indxTrain = list(filter(lambda x: x not in indxTest and x not in indxVal, indices)) #al no meter clases con un solo patron al indxTest, pasan automaticamente al indxTrain
    return indxTrain, indxTest,indxVal

#/////////////////////////////////////////////////////////////

def FindMaxLength(lst): 
    maxLength = max(len(x) for x in lst ) 
  
    return maxLength

#/////////////////////////////////////////////////////////////

def embeddear(indx, dataset, wordSet, tensorEmbedding):
    cant_patrones = len(indx)
    tensores = []
    prom_tensor = np.zeros((cant_patrones,300))
    dataset = adaptarDataset(dataset) #esto pasa de vectores tipo ['holi' 'perri'] a [['holi'] ['perri']] que es lo que necesito
    for i in range(cant_patrones):
        words = word_tokenize(dataset[indx[i],0]) #Tengo las palabras de una oración
        cant_palabras = len(words)
        prom = 0
        tensores_locales = []
        for w in words:
            tensor = tensorEmbedding[wordSet[w]]
            tensores_locales.append(tensor)
            prom += tensor
        prom_tensor[i] = prom/cant_palabras
        tensores.append(tensores_locales)
    maxlen = FindMaxLength(tensores)
    return tensores, prom_tensor, maxlen

#/////////////////////////////////////////////////////////////

def longest(l):
    if(not isinstance(l, list)): return(0)
    return(max([len(l),] + [len(subl) for subl in l if isinstance(subl, list)] +
        [longest(subl) for subl in l]))

#/////////////////////////////////////////////////////////////

def formatearTensores(tensores,maxlen,sizeTensor=300):
    #Necesito meter todo en un array de cant_patrones * maxlen * 300
    #maxlen = longest(tensores)
    print("Longitud maxima para hardcodear: ",maxlen)
    cant_patrones = len(tensores)
    
    #Inicializo....
    tensorMatrix = np.zeros((cant_patrones, maxlen, sizeTensor), dtype=np.float32)
    
    #Armo la matriz necesaria de patrones*maxlen*300
    for i in range(cant_patrones):
        for j in range(len(tensores[i])):
            tensorMatrix[i][j] = tensores[i][j]
        #print(tensorMatrix[i])
    tensorMatrix = torch.from_numpy(tensorMatrix)
    return tensorMatrix 


#/////////////////////////////////////////////////////////////

def getLabels(indxTrain, indxTest, indxVal, dataset):
    Ytrain = []
    Ytest = []
    Yval = []
    for i in range(dataset.shape[0]):
        if i in indxTrain:
            Ytrain.append(dataset[i][0])
        if i in indxTest:
            Ytest.append(dataset[i][0])
        if i in indxVal:
            Yval.append(dataset[i][0])

    #Convirtiendo formatos de labels a tensores...

    Ytrain = torch.FloatTensor(Ytrain)
    Ytest = torch.FloatTensor(Ytest)
    Yval = torch.FloatTensor(Yval)

    return Ytrain, Ytest, Yval

#/////////////////////////////////////////////////////////////

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,maxlen,num_layers):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        if num_layers == 2:
            self.block=nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2))
        elif num_layers == 3:
            self.block=nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2))
        elif num_layers == 4:
            self.block=nn.Sequential(nn.Conv1d(input_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2),
            nn.Conv1d(hidden_size,hidden_size,3,padding=1),nn.ReLU(),nn.BatchNorm1d(hidden_size),nn.AvgPool1d(2))
        self.out = nn.Linear(hidden_size*int(maxlen/2**num_layers),output_size)

    def forward(self, inputs, hidden=None):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        #inputs = inputs.transpose(0, 1).transpose(1, 2)

        #entra un tensor de dimensiones: batch x seq_len x input_size
        #quiero un tensor dedimensiones: batch x input_size x seq_len
        inputs = inputs.transpose(1, 2)
        x = self.block(inputs)
    
        p = torch.flatten(x,start_dim=1)
        output = self.out(p)

        return output

cantidad = 100
path_vectors = 'C:/Users/lucy/chatbot/fasttext-sbwc.3.6.e20.vec'
path_dataset = "C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv"
wordvectors = KeyedVectors.load_word2vec_format(path_vectors, limit=cantidad)
dataset = pd.read_csv(path_dataset,delimiter=',',header=None)
cant_preg = dataset.values.shape[0]
cantidad_labels = dataset.values[len(dataset.values)-1,0] + 1
#Genero el diccionario del dataset completo

_,tensorEmbedding,wordSet = generarDiccionario(dataset.values,wordvectors)
print(tensorEmbedding.shape)

tensores,prom_tensor,maxlen = embeddear(range(cant_preg),dataset.values,wordSet,tensorEmbedding)

X = formatearTensores(tensores,maxlen)

Y = [x[0] for x in dataset.values]
from collections import Counter
count = Counter(Y)
print("clase y cantidad de patrones: ",sorted(count.items(), key=lambda pair: pair[1], reverse=True))
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    #device = torch.device("cuda")
    device = 'cuda'
    print("GPU is available")
else:
    device = 'cpu'
    #device = torch.device("cpu")
    print("GPU not available, CPU used")

dict_size = len(X[0][0])
print("Dict_size: ",dict_size)
print("cantidad d labels: ",cantidad_labels)

n_epochs = 500
batch_size = 500
hidden_size = 20
num_layers=3
model = CNN(input_size=dict_size,hidden_size = hidden_size,output_size=cantidad_labels,maxlen=maxlen,num_layers=num_layers)

# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

parameters = {    
    #'module__hidden_dim' : [12,24,48,96],
    'module__hidden_size' : [20,30,40],    
    #'max_epochs' : [50,70,90,110,130,150]
    'module__num_layers':[2,3,4],
    'max_epochs' : [1],
    'batch_size' : [1000,500,700]
}

candidatos = 5
net = NeuralNetClassifier(model,module__input_size = dict_size,module__hidden_size=hidden_size,module__output_size=cantidad_labels,module__maxlen=maxlen,module__num_layers=num_layers,criterion=torch.nn.CrossEntropyLoss,optimizer=torch.optim.Adam,verbose=1,device=device)
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=candidatos)
X_train,X_test,y_train,y_test = train_test_split(X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)
gs.fit(X_train,y_train_tensor)

hidden_size = []
max_epochs = []
batch_size = []
num_layers = []
score = []
std = []

# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        print(i)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            batch_size.append(results['params'][candidate]['batch_size'])
            hidden_size.append(results['params'][candidate]['module__hidden_size'])
            max_epochs.append(results['params'][candidate]['max_epochs'])
            num_layers.append(results['params'][candidate]['module__num_layers'])
            score.append(results['mean_test_score'][candidate])
            std.append(results['std_test_score'][candidate])

report(gs.cv_results_,candidatos)
ejex = batch_size
ejey = num_layers
print(ejey)
ejez = score
plotx,ploty, = np.meshgrid(np.linspace(np.min(ejex),np.max(ejex),10),\
                           np.linspace(np.min(ejey),np.max(ejey),10))
plotz = interp.griddata((ejex,ejey),ejez,(plotx,ploty),method='linear')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')

probs = gs.best_estimator_.predict_proba(X_test)

# get training and validation loss
epochs = [i for i in range(len(gs.best_estimator_.history))]
train_loss = gs.best_estimator_.history[:,'train_loss']
valid_loss = gs.best_estimator_.history[:,'valid_loss']
acc = balanced_accuracy_score(y_test_tensor,np.argmax(probs,axis=1))
print("tasa de acierto obtenida: ",acc)
fig1 = plt.figure()
plt.plot(epochs,train_loss,'g-')
plt.plot(epochs,valid_loss,'r-')
plt.title('Training Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend(['Train','Validation'])

plt.show()
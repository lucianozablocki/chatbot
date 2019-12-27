from gensim.models.keyedvectors import KeyedVectors
#import gensim
#import pandas as pn
from nltk import word_tokenize
import numpy as np
import preprocesamiento
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
import utils
from utils import EarlyStopping
import skorch
from skorch import NeuralNetClassifier,NeuralNet
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

class MLP(nn.Module):
    def __init__(self,D_in,arq,D_out):
        #constructor: aca se instancian dos modulos del tipo nn.Linear
        #y las asignamos como variables linear1 y linear 2
        super(MLP, self).__init__()
        #self.arq = nn.ParameterList(arq)
        self.arq = arq
        self.D_in = D_in
        self.D_out = D_out
        for index in range(len(arq)):
          self.parameters = nn.Parameter(self.arq[index])
        if arq[0] == 1:
            self.block = nn.Sequential(nn.Linear(D_in,arq[1][0]),nn.ReLU(),nn.Linear(arq[1][0],D_out))
        if arq[0] == 2:
            self.block = nn.Sequential(nn.Linear(D_in,arq[1][0]),nn.ReLU(),nn.Linear(arq[1][0],arq[1][1]),nn.ReLU(),nn.Linear(arq[1][1],D_out))
        elif arq[0] == 3:
            self.block = nn.Sequential(nn.Linear(D_in,arq[1][0]),nn.ReLU(),nn.Linear(arq[1][0],arq[1][1]),nn.ReLU(),
            nn.Linear(arq[1][1],arq[1][2]),nn.ReLU(),nn.Linear(arq[1,2],D_out))
        #self.linear1 = nn.Linear(D_in,H)
        #self.linear2 = nn.Linear(H,D_out)

    def forward(self, x):
        #definicion del paso hacia adelante, entra un tensor y devuelve otro tensor 
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        """
        #y_pred = F.softmax(self.linear2(h_relu)) #Aplico la softmax -> no es necesario (incluida en crossentropy)
        y_pred = self.block(x)
        return y_pred

#def train_prueba2(path_vectors,path_dataset,batch_size,epocas,patience=None):
#cargar vectores de embedding
print("----------CARGANDO VECTORES DE EMBEDDING--------------")
#wordvectors_file_vec = path_vectors
path_vectors = 'C:/Users/lucy/chatbot/fasttext-sbwc.3.6.e20.vec'
path_dataset = "C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv"
cantidad = 100
wordvectors = KeyedVectors.load_word2vec_format(path_vectors, limit=cantidad)
print("----------FINISHED CARGANDO VECTORES DE EMBEDDING--------------\n")

#cargar csv con preguntas preprocesadas
print("-----------CARGANDO/CREANDO PREGUNTAS PREPROCESADAS--------------")
#correctedData = preprocesamiento.preprocesar(dataset.values,1) #Dataset lematizado, descomentar en caso de crear otro dataset 
correctedData = pd.read_csv(path_dataset,delimiter=',') #comentar esta linea en caso de descomentar la anterior
print("----------FINISHED CARGANDO/CREANDO PREGUNTAS PREPROCESADAS-------------\n")

#obtener datos utiles sobre el dataset
labels = correctedData.values[:,0]
print("Shape de corrected data: ",correctedData.shape)
cantidad_labels = correctedData.values[len(correctedData.values)-1,0] + 1
cantidad_preg = correctedData.shape[0]
print("Cantidad de clases: ",cantidad_labels)
print("Cantidad de patrones: ",cantidad_preg)
print("Lista de clases para cada pregunta :",labels)
correctedData = correctedData.values

print("-------------CREANDO INDICES DE TEST Y TRAIN------------------")
"""
indxTest,indxTrain,indxVal = utils.separate_dataset(correctedData,cantidad_preg)
cant_test = len(indxTest)
cant_train = len(indxTrain)
cant_val = len(indxVal)

print("cantidad de patrones de prueba: ",cant_test)
print("cantidad de patrones de entrenamiento: ",cant_train)
print("cantidad de patrones de validacion: ",cant_val)
print("-------------FINISHED CREANDO INDICES DE TEST Y TRAIN------------------\n")
"""
print("-------------CREANDO Ytest e Ytrain (groundtruth de cada subconjunto), Xtext_test y Xtext_train-------------------")
"""
#assert cant_test + cant_train == cantidad_preg 
#Ytrain = np.zeros((cant_train,1), dtype=np.int64)
Ytrain = np.zeros((cant_train),dtype=np.int64) #clases de los patrones que estan en el subconjunto de train
Xtext_train = np.zeros((cant_train,1), dtype=object)
#Ytest = np.zeros((cant_test,1), dtype=np.int64)
Ytest = np.zeros((cant_test), dtype=np.int64) #clases de los patrones que estan en el subconjunto de test
Xtext_test = np.zeros((cant_test,1), dtype=object)

Yval = np.zeros((cant_val), dtype=np.int64) #clases de los patrones que estan en el subconjunto de validacion
Xtext_val = np.zeros((cant_val,1), dtype=object)

contTrain = 0
contTest = 0
contVal = 0

for i in range(cantidad_preg):
    if i in indxTrain:
        Ytrain[contTrain] = correctedData[i,0]
        Xtext_train[contTrain] = correctedData[i,1]
        contTrain+=1
    elif i in indxTest:
        Ytest[contTest] = correctedData[i,0]
        Xtext_test[contTest] = correctedData[i,1]
        contTest+=1
    if i in indxVal:
        Yval[contVal] = correctedData[i,0]
        Xtext_val[contVal] = correctedData[i,1]
        contVal+=1
"""
X_text = np.zeros((cantidad_preg,1), dtype=object)
Y = np.zeros((cantidad_preg),dtype=np.int64)
for i in range(cantidad_preg):
    X_text[i] = correctedData[i,1]
    Y[i] = correctedData[i,0]

Y = torch.from_numpy(Y)
print(Y)
"""
Ytrain = torch.from_numpy(Ytrain)
print("shape del tensor Ytrain : ",Ytrain.shape)
Ytest = torch.from_numpy(Ytest)
print("shape del tensor Ytest : ",Ytest.shape)
Yval = torch.from_numpy(Yval)
print("shape del tensor Yval : ",Yval.shape)
print(Xtext_val.shape)
"""
print("-------------FINISHED CREANDO Ytest, Ytrain, Xtext_test y Xtext_train-------------------\n")
parameters = {    
'hidden_layer_sizes' : [(200),(100),(200,100),(250,200,100)],
#'n_layers' : [1,2,3,4],
#'dropout' : [0.2,0.3,0.4,0.5,0.6],
#'bidirectional' : [True,False],
'max_iter' : [1,2,3,4,5],
'alpha' : [0.00001,0.0001,0.001,0.1,0.2],
'batch_size' : [100,200,300]
}
D_in = 300
arq = [[2],[200,100]]
#rint(arq[0])
#print(arq[1])
#print(arq[0][1][0])

D_out = cantidad_labels
X,_,_ = utils.create_tensor_prom_and_embedding(X_text,cantidad_preg,wordvectors)
#seq = nn.Sequential(nn.Linear(D_in,H),nn.ReLU(),nn.Linear(H,D_out))
#model = MLP(D_in,arq,D_out)
net = MLPClassifier(hidden_layer_sizes=(D_in,100,D_out))
#net = NeuralNetClassifier(module=model,module__hidden_layer_sizes=(D_in,100,D_out),verbose=1)
print(net.get_params().keys())
candidatos = 10
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=candidatos)
X_train,X_test,y_train,y_test = train_test_split(X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
gs.fit(X_train,y_train)
# Utility function to report best scores (found online)
hidden_layer_sizes = []
max_epochs = []
alpha= []
batch_size=[]
std=[]
score=[]
y_test_tensor = torch.LongTensor(y_test)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            score.append(results['mean_test_score'][candidate])
            std.append(results['std_test_score'][candidate])
            hidden_layer_sizes.append(results['params'][candidate]['hidden_layer_sizes'])
            batch_size.append(results['params'][candidate]['batch_size'])
            alpha.append(results['params'][candidate]['alpha'])
            max_epochs.append(results['params'][candidate]['max_iter'])

report(gs.cv_results_,candidatos)            

ejex = batch_size
ejey = max_epochs
#print(ejez)
#print(len(ejez))
#ejez = [0.5,0.3,0.4,0.33,0.6,0.45,0.75,0.8,0.2,0.47,0.56,0.66,0.9,0.87,0.67,0.43]
ejez = score
plotx,ploty, = np.meshgrid(np.linspace(np.min(ejex),np.max(ejex),10),\
                           np.linspace(np.min(ejey),np.max(ejey),10))
plotz = interp.griddata((ejex,ejey),ejez,(plotx,ploty),method='linear')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')

#print(X_test.shape)            
probs = gs.best_estimator_.predict_proba(X_test)
#print(probs.shape)

# get training and validation loss
#epochs = [i for i in range(len(gs.best_estimator_.history))]
epochs = [i for i in range(max_epochs[0])]
#train_loss = gs.best_estimator_.history[:,'train_loss']
train_loss = gs.best_estimator_.loss_curve_

#valid_loss = gs.best_estimator_.history[:,'valid_loss']
acc = balanced_accuracy_score(y_test_tensor,np.argmax(probs,axis=1))
print("tasa de acierto obtenida: ",acc)
fig1 = plt.figure()
plt.plot(epochs,train_loss,'g-')
#plt.plot(epochs,valid_loss,'r-')
plt.title('Training Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend(['Train','Validation'])

plt.show()

"""
print("--------------CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA ENTRENAMIENTO Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------")
Xtrain,tensorEmbeddingTrain,wordSetTrain = utils.create_tensor_prom_and_embedding(Xtext_train,cant_train,wordvectors)
print("Size de tensor embedding train: ",tensorEmbeddingTrain.shape)
print("Size de Xtrain: ", Xtrain.shape)    
print("--------------FINISHED CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA ENTRENAMIENTO Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------\n")

print("--------------CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA VALIDACION Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------")
Xval,tensorEmbeddingVal,wordSetVal = utils.create_tensor_prom_and_embedding(Xtext_val,cant_val,wordvectors)
print("Size de tensor embedding test: ",tensorEmbeddingVal.shape)
print("Size de Xtest: ", Xval.shape)    
print("--------------FINISHED CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA VALIDACION Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------\n")

print("--------------CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA TEST Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------")
Xtest,tensorEmbeddingTest,wordSetTest = utils.create_tensor_prom_and_embedding(Xtext_test,cant_test,wordvectors)
print("Size de tensor embedding test: ",tensorEmbeddingTest.shape)
print("Size de Xtest: ", Xtest.shape)    
print("--------------FINISHED CREANDO TENSOR DE PROMEDIOS DE VECTORES DE EMBEDDING VOCABULARIO ORIGINAL PARA TEST Y TENSOR DE EMBEDDINGS VOCABULARIO PROPIO----------------------\n")
"""
    
print("-----------------INIT TRAINING PROCESS---------------------")
"""
D_in = 300
H = 200
D_out = cantidad_labels
#Creo el trainset y el testset
trainset = torch.utils.data.TensorDataset(Xtrain,Ytrain) 
testset = torch.utils.data.TensorDataset(Xtest,Ytest)
valset = torch.utils.data.TensorDataset(Xval,Yval)
#Creo el dataloader
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True) 
testloader = torch.utils.data.DataLoader(testset,batch_size=cant_test,shuffle=True)
valloader = torch.utils.data.DataLoader(valset,batch_size=cant_val,shuffle=True)

prueba2 = MLP(D_in,H,D_out)
criterion = nn.CrossEntropyLoss(reduction = "sum") #analizar reduction
optimizerPrueba2 = optim.Adam(prueba2.parameters())
early_stopping = EarlyStopping(verbose=True,patience=patience)
train_losses = [] #vector que guarda el loss para cada epoca
val_losses = []
avg_train_losses = []
avg_val_losses = []
ejex = []
print("------------- Entrenando un MLP con " , D_in ," neuronas de entrada, " , H , " neuronas en la capa oculta y " , D_out , " neuronas de salida-----------------" )
for t in range(epocas): 
    for batch_idx, (XX, YY) in enumerate(trainloader):
        out = prueba2(XX)
        loss = criterion(out, YY)
        optimizerPrueba2.zero_grad()
        loss.backward()
        optimizerPrueba2.step()
        train_losses.append(loss.item())


    for (XX,YY) in valloader:
        out = prueba2(XX)
        loss = criterion(out,YY)
        val_losses.append(loss.item())

    train_loss = np.average(train_losses)
    val_loss = np.average(val_losses)
    avg_train_losses.append(train_loss)
    avg_val_losses.append(val_loss)
    ejex.append(t)
    epoch_len = len(str(epocas))
    
    print_msg = (f'[{t:>{epoch_len}}/{epocas:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {val_loss:.5f}')
    print(print_msg)
    
    train_losses = []
    val_losses = []

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(val_loss, prueba2)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
prueba2.load_state_dict(torch.load('checkpoint.pt'))
print("-------------------------------FINISHED TRAINING PROCESS-----------------------------\n")

print("--------------------------------INIT TEST PROCESS----------------------------")
acc=0
for batch_idx, (XX,YY) in enumerate(testloader):
    out = prueba2(XX)
    for i in range(cant_test):
        valor, outNet = torch.max(out[i],0)
        #print("Salida de la red: ",outNet)
        #print("La posta: ",YY[i].item())
        if outNet == YY.numpy()[i]:
            acc+=1
acc = acc/cant_test*100
print("------------------------------FINISHED TEST PROCESS-----------------------------------")

return ejex,acc,avg_train_losses,avg_val_losses
"""
"""
if __name__ == '__main__':
    #batch_size = 5
    #epocas = 200
    path_vectors = 'C:/Users/lucy/chatbot/fasttext-sbwc.3.6.e20.vec'
    path_dataset = "C:/Users/lucy/chatbot/preprocessedQuestions_lem.csv"
    ejex,acc,trainloss,valloss = train_prueba2(path_vectors,path_dataset,batch_size,epocas,patience = 20)
    # find position of lowest validation loss
    minposs = valloss.index(min(valloss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    print("Tasa de acierto obtenida: ", acc)
    plt.plot(ejex,trainloss)
    plt.plot(ejex,valloss)
    plt.show()
"""
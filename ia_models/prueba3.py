import numpy as np
import torch
import matplotlib.pyplot as plt
#from torchvision import datasets
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from utils import EarlyStopping
import bow
from bow import BOW
from nltk.corpus import stopwords
#from prueba2 import MLP
import skorch
from skorch import NeuralNetClassifier,NeuralNet
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

class MLP_3ocultas(nn.Module):
    def __init__(self,D_in,H1,H2,H3,D_out):
        #constructor: aca se instancian dos modulos del tipo nn.Linear
        #y las asignamos como variables linear1 y linear 2
        super(MLP_3ocultas, self).__init__()
        self.linear1 = nn.Linear(D_in,H1)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,H3)
        self.linear4 = nn.Linear(H3,D_out)

    def forward(self, x):
        #definicion del paso hacia adelante, entra un tensor y devuelve otro tensor 
        h_relu = self.linear1(x).clamp(min=0)
        h = self.linear2(h_relu)
        h1 = self.linear3(h)
        y_pred = self.linear4(h1)
        #y_pred_sigmoid = torch.sigmoid(y_pred)
        #y_pred = F.softmax(self.linear2(h_relu)) #Aplico la softmax -> no es necesario (incluida en crossentropy)
        
        return y_pred

#cargar csv con preguntas preprocesadas
print("-----------CARGANDO/CREANDO PREGUNTAS PREPROCESADAS--------------")
#correctedData = preprocesamiento.preprocesar(dataset.values,1) #Dataset lematizado, descomentar en caso de crear otro dataset 
correctedData = pd.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv",delimiter=',') #comentar esta linea en caso de descomentar la anterior
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
#indxTest,indxTrain,indxVal = utils.separate_dataset(correctedData,cantidad_preg)
#cant_test = len(indxTest)
#cant_train = len(indxTrain)
#cant_val = len(indxVal)
#print("cantidad de patrones de prueba: ",cant_test)
#print("cantidad de patrones de entrenamiento: ",cant_train)
#print("cantidad de patrones de validacion: ",cant_val)
print("-------------FINISHED CREANDO INDICES DE TEST Y TRAIN------------------\n")
stoplist = stopwords.words('spanish')
text = correctedData[:,1]
bow_unigram = BOW(text,'ascii',stoplist,weighting=True)

print("-------------CREANDO Ytest e Ytrain (groundtruth de cada subconjunto), Xtext_test y Xtext_train-------------------")
Y = np.zeros((cantidad_preg),dtype=np.int64)
for i in range(cantidad_preg):
    Y[i] = correctedData[i,0]

Y = torch.from_numpy(Y)
print(Y)
"""
#assert cant_test + cant_train == cantidad_preg 
#Ytrain = np.zeros((cant_train,1), dtype=np.int64)
Ytrain = np.zeros((cant_train),dtype=np.int64) #clases de los patrones que estan en el subconjunto de train
Xtrain = np.zeros((cant_train, bow_unigram.X.shape[1]), dtype=np.float) #opcion p mejorar performance, inicializar como sparse cada matriz X
#Ytest = np.zeros((cant_test,1), dtype=np.int64)
Ytest = np.zeros((cant_test), dtype=np.int64) #clases de los patrones que estan en el subconjunto de test
Xtest = np.zeros((cant_test, bow_unigram.X.shape[1]), dtype=np.float)

Yval = np.zeros((cant_val), dtype=np.int64) #clases de los patrones que estan en el subconjunto de validacion
Xval = np.zeros((cant_val, bow_unigram.X.shape[1]), dtype=np.float)

contTrain = 0
contTest = 0
contVal = 0
#print(bow_unigram.X.shape)
#print(Xtrain.shape)
for i in range(cantidad_preg):
    if i in indxTrain:
        Ytrain[contTrain] = correctedData[i,0]
        Xtrain[contTrain] = bow_unigram.X.getrow(i).todense()
        contTrain+=1
    elif i in indxTest:
        Ytest[contTest] = correctedData[i,0]
        Xtest[contTest,:] = bow_unigram.X.getrow(i).todense()
        contTest+=1
    if i in indxVal:
        Yval[contVal] = correctedData[i,0]
        Xval[contVal,:] = bow_unigram.X.getrow(i).todense()
        contVal+=1


Ytrain = torch.from_numpy(Ytrain)
print("shape del tensor Ytrain : ",Ytrain.shape)
Ytest = torch.from_numpy(Ytest)
print("shape del tensor Ytest : ",Ytest.shape)
Yval = torch.from_numpy(Yval)
print("shape del tensor Yval : ",Yval.shape)
Xtrain = torch.from_numpy(Xtrain)
Xtest = torch.from_numpy(Xtest)
Xval = torch.from_numpy(Xval)
Xtrain = Xtrain.float()
Xtest = Xtest.float()
Xval = Xval.float()
"""
print("-------------FINISHED CREANDO Ytest, Ytrain, Xtext_test y Xtext_train-------------------\n")

print("-----------------INIT TRAINING PROCESS---------------------")
parameters = {    
'hidden_layer_sizes' : [(200),(100),(200,100),(250,200,100)],
#'n_layers' : [1,2,3,4],
#'dropout' : [0.2,0.3,0.4,0.5,0.6],
#'bidirectional' : [True,False],
'max_iter' : [1,2,3,4,5],
'alpha' : [0.00001,0.0001,0.001,0.1,0.2],
'batch_size' : [100,200,300]
}

D_in = bow_unigram.X.shape[1]
D_out = cantidad_labels

net = MLPClassifier(hidden_layer_sizes=(D_in,100,D_out))
candidatos = 10
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=candidatos)
X_train,X_test,y_train,y_test = train_test_split(bow_unigram.X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)

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
#Creo el trainset y el testset
#print(Xtrain.shape)
trainset = torch.utils.data.TensorDataset(Xtrain,Ytrain) 
testset = torch.utils.data.TensorDataset(Xtest,Ytest)
valset = torch.utils.data.TensorDataset(Xval,Yval)
batch_size = Xtrain.shape[0]
epocas = 200
patience = 50
#Creo el dataloader
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True) 
testloader = torch.utils.data.DataLoader(testset,batch_size=cant_test,shuffle=True)
valloader = torch.utils.data.DataLoader(valset,batch_size=cant_val,shuffle=True)

prueba3 = MLP_3ocultas(D_in,H1,H2,H3,D_out)
criterion = nn.CrossEntropyLoss(reduction = "sum") #analizar reduction
optimizerPrueba3 = optim.Adam(prueba3.parameters())
early_stopping = EarlyStopping(verbose=True,patience=patience)
train_losses = [] #vector que guarda el loss para cada epoca
val_losses = []
avg_train_losses = []
avg_val_losses = []
ejex = []
print("------------- Entrenando un MLP con " , D_in ," neuronas de entrada, " , H1 ,H2,H3, " neuronas en la capa oculta y " , D_out , " neuronas de salida-----------------" )
for t in range(epocas): 
    for batch_idx, (XX, YY) in enumerate(trainloader):
        out = prueba3(XX)
        loss = criterion(out, YY)
        optimizerPrueba3.zero_grad()
        loss.backward()
        optimizerPrueba3.step()
        train_losses.append(loss.item())


    for (XX,YY) in valloader:
        out = prueba3(XX)
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
    early_stopping(val_loss, prueba3)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
prueba3.load_state_dict(torch.load('checkpoint.pt'))
print("-------------------------------FINISHED TRAINING PROCESS-----------------------------\n")

print("--------------------------------INIT TEST PROCESS----------------------------")
acc=0
for batch_idx, (XX,YY) in enumerate(testloader):
    out = prueba3(XX)
    for i in range(cant_test):
        valor, outNet = torch.max(out[i],0)
        #print("Salida de la red: ",outNet)
        #print("La posta: ",YY[i].item())
        if outNet == YY.numpy()[i]:
            acc+=1
acc = acc/cant_test*100
print(acc)
print("------------------------------FINISHED TEST PROCESS-----------------------------------")

"""
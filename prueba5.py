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

class sRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, dropout,bidirectional = False):
        super(sRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout = dropout,bidirectional = bidirectional)   
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True) 
        # Fully connected layer
        self.pool = nn.AvgPool2d(2)
        #self.fc = nn.Linear(hidden_dim*maxlen*2, output_size)
        print("maxlen: ",maxlen)
        self.fc = nn.Linear(int((hidden_dim*maxlen)*1/2), output_size)
        self.fc2 = nn.Linear(output_size,output_size)
    
    def forward(self, x):
        
        #batch_size = x.size(0)
        # print("Batch_size en la capa Forward: ",batch_size)
        #print("este es el batch_size: ",x.size(0))
        #Initializing hidden state for first input using method defined below
        # print("La paso a la hidden...")
        #hidden = self.init_hidden(batch_size)
        # print("Salí de la hidden, entro a la out...")
        # Passing in the input and hidden state into the model and obtaining outputs
        #out, hidden = self.rnn(x, hidden)
        out,_ = self.rnn(x)
        # print("Salí de la out...")
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        #print("shape de out q entra a la fully connected ",out.shape)
        #import ipdb;ipdb.set_trace()
        #print(out.shape)
        out = self.pool(out)
        #print(out.shape)
        out = self.fc(out.reshape(-1,out.shape[2]*out.shape[1]))
        out = self.fc2(out)
        #out = self.fc2(out)
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

#/////////////////////////////////////////////////////////////

#cantidad = 100000
cantidad = 100
path_vectors = 'C:/Users/lucy/chatbot/fasttext-sbwc.3.6.e20.vec'
path_dataset = "C:/Users/lucy/chatbot/preprocessedQuestions_lem.csv"
wordvectors = KeyedVectors.load_word2vec_format(path_vectors, limit=cantidad)
dataset = pd.read_csv(path_dataset,delimiter=',',header=None)
cant_preg = dataset.values.shape[0]
#Genero el diccionario del dataset completo

X,tensorEmbedding,wordSet = generarDiccionario(dataset.values,wordvectors)

#Genero los indices que necesito:

#indxTrain, indxTest,indxVal = separate_dataset(dataset.values)

#Genero los tensores de train, test y val

#tensoresTest, prom_tensor_Test,maxlenTest = embeddear(indxTest,dataset.values,wordSet,tensorEmbedding)
#tensoresTrain, prom_tensor_Train, maxlenTrain = embeddear(indxTrain,dataset.values,wordSet,tensorEmbedding)
#tensoresVal, prom_tensor_Val, maxlenVal = embeddear(indxVal,dataset.values,wordSet,tensorEmbedding)
tensores,prom_tensor,maxlen = embeddear(range(cant_preg),dataset.values,wordSet,tensorEmbedding)
#maxlen = max(maxlenTest,maxlenTrain,maxlenVal)

#Y acá genero las entradas a la LSTM

#Xtrain = formatearTensores(tensoresTrain,maxlen)
#Xtest = formatearTensores(tensoresTest,maxlen)
#Xval = formatearTensores(tensoresVal,maxlen)
X = formatearTensores(tensores,maxlen)
# Genero los tensores de Labels

#Ytrain, Ytest, Yval = getLabels(indxTrain,indxTest,indxVal,dataset.values)
Y = [x[0] for x in dataset.values]

#Cuestiones propias de la red recurrente....

#batch_size_train = len(Xtrain)
#batch_size_test = len(Xtest)
#batch_size_val = len(Xval)
#seq_len = maxlen
#input_seq_train = Xtrain
#input_seq_test = Xtest
#input_seq_val = Xval
# input_seq_train = torch.from_numpy(input_seq_train)
# input_seq_test = torch.from_numpy(input_seq_test)
#target_seq_train = Ytrain
#target_seq_test = Ytest
#target_seq_val = Yval
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
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

cantidad_labels = dataset.values[len(dataset.values)-1,0] + 1
# Instantiate the model with hyperparamet-ers

dict_size = len(X[0][0])
print("Dict_size: ",dict_size)
hidden_dim=24
n_layers=2
dropout=0.5
bidirectional=True
n_epochs = 500
model = sRNN(input_size=dict_size, output_size=cantidad_labels, hidden_dim=hidden_dim, num_layers=n_layers, dropout =dropout,bidirectional = bidirectional)

# We'll also set the model to the device that we defined earlier (default is CPU)
#model = model.to(device)

# Define hyperparameters

batch_size = 500
dropout= 0.5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
#lr=0.01
#Creo el trainset y el testset
#trainset = torch.utils.data.TensorDataset(input_seq_train,Ytrain) 
#testset = torch.utils.data.TensorDataset(input_seq_test,Ytest)
#valset = torch.utils.data.TensorDataset(input_seq_val,Yval)

parameters = {    
    #'module__hidden_dim' : [12,24,48,96],
    'module__hidden_dim' : [24],    
    #'module__num_layers' : [1,2,3,4],
    'module__num_layers' : [2],
    #'module__dropout' : [0.2,0.3,0.4,0.5,0.6],
    'module__dropout' : [0.5,0.2,0.7,0.4],
    #'max_epochs' : [50,70,90,110,130,150]
    'max_epochs' : [1],
    'batch_size' : [500,1000,300,100]
}

#cant_test = batch_size_test
#cant_val = batch_size_val
#module__input_size=dict_size,module__output_size=dict_size,module__hidden_dim=hidden_dim,module__n_layers=n_layers, module__dropout =dropout,module__bidirectional = bidirectional,
#,criterion=torch.nn.CrossEntropyLoss(),optimizer=torch.optim.Adam(),verbose=1
net = NeuralNetClassifier(model,module__input_size = dict_size,module__output_size=cantidad_labels,module__hidden_dim=hidden_dim,module__num_layers=n_layers, module__dropout =dropout,module__bidirectional = bidirectional,criterion=torch.nn.CrossEntropyLoss,optimizer=torch.optim.Adam,verbose=1,device=device)
#print(net.get_params().keys())
#net = NeuralNetClassifier(module=seq,criterion=torch.nn.CrossEntropyLoss(),optimizer=torch.optim.Adam(),verbose=1)
#gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=1)
gs = GridSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy')
X_train,X_test,y_train,y_test = train_test_split(X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
#print(X_train.shape)
#print(type(X_train))
#print(y_train.shape)
#print(type(y_train))
#print(X_test.shape)
#print(type(X_test))
#print(y_train.shape)
#print(type(y_train))
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)
gs.fit(X_train,y_train_tensor)
hidden_dim=[]
num_layers=[]
dropout=[]
max_epochs=[]
batch_size=[]
score=[]
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
            dropout.append(results['params'][candidate]['module__dropout'])
            batch_size.append(results['params'][candidate]['batch_size'])
            hidden_dim.append(results['params'][candidate]['module__hidden_dim'])
            max_epochs.append(results['params'][candidate]['max_epochs'])
            num_layers.append(results['params'][candidate]['module__num_layers'])
            score.append(results['mean_test_score'][candidate])

report(gs.cv_results_,16)
ejex = batch_size
ejey = dropout
#print(ejez)
#print(len(ejez))
ejez = [0.5,0.3,0.4,0.33,0.6,0.45,0.75,0.8,0.2,0.47,0.56,0.66,0.9,0.87,0.67,0.43]
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

"""
#Creo el dataloader
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True) 
testloader = torch.utils.data.DataLoader(testset,batch_size=cant_test,shuffle=True)
valloader = torch.utils.data.DataLoader(valset,batch_size=cant_val,shuffle=True)
patience=7
early_stopping = EarlyStopping(verbose=True,patience=patience)
train_losses = [] #vector que guarda el loss para cada epoca
val_losses = []
avg_train_losses = []
avg_val_losses = []
ejex = []
# Define Loss, Optimizer

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

early_stopping = EarlyStopping(verbose=True,patience=patience)
input_seq_train = input_seq_train.to(device)
# target_prob_train = torch.from_numpy(target_prob_train)

input_seq_test= input_seq_test.to(device)
# target_prob_test = torch.from_numpy(target_prob_test)
print("input_seq_train.shape: ",input_seq_train.shape)
"""
"""
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq_train)
    output = output.to(device)
    #target_seq = target_prob_train.to(device)
    target_seq = Ytrain.to(device)
    # print("Target_seq: ",target_seq.shape)
    # print("shape de la salida: ",output.shape)
    #print("shape target: ",target_seq.shape)
    #loss = criterion(output, target_seq.view(-1).long())
    loss = criterion(output, target_seq.long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

output, hidden = model(input_seq_test)

#print("Salida de la red: ",outNet)
#print("La posta: ",YY[i].item())
#acc = 0
print("SHAPE DE LA SALIDA DE LA RED: ",output.shape)
"""
"""
for t in range(n_epochs): 
    for batch_idx, (XX, YY) in enumerate(trainloader):
        
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        #input_seq = input_seq.to(device)
        output, hidden = model(XX)
        output = output.to(device)
        #target_seq = target_prob_train.to(device)
        target_seq = YY.to(device)
        loss = criterion(output, target_seq.long())
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        train_losses.append(loss.item())      

    for (XX,YY) in valloader:
        out,_= model(XX)
        YY = YY.to(device)
        loss = criterion(out,YY.long())
        val_losses.append(loss.item())

    train_loss = np.average(train_losses)
    val_loss = np.average(val_losses)
    avg_train_losses.append(train_loss)
    avg_val_losses.append(val_loss)
    ejex.append(t)
    epoch_len = len(str(n_epochs))
    
    print_msg = (f'[{t:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {val_loss:.5f}')
    print(print_msg)
    
    train_losses = []
    val_losses = []

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

output, hidden = model(input_seq_test)
acc = balanced_accuracy_score(Ytest,output.argmax(dim=1))
minposs = avg_val_losses.index(min(avg_val_losses))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
print("Tasa de acierto obtenida: ", acc)
plt.plot(ejex,avg_train_losses)
plt.plot(ejex,avg_val_losses)
plt.show()
"""
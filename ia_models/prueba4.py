import torch
from torch import nn
import numpy as np
import pandas as pn
from char_tokenizer import CharTokenizer
import utils
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F
from utils import EarlyStopping
from matplotlib import pyplot as plt
import skorch
from skorch import NeuralNetClassifier,NeuralNet
from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
def one_hot_encode(indx, dict_size, maxlen, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, maxlen, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(maxlen):
            features[i, u, int(indx[i][u])] = 1
    return features

class sRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,bidirectional=False):
        super(sRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True,bidirectional=bidirectional)   
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True) 
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*169, output_size) # *2 agregado xq hay el doble de neuronas en bidirectional
        #self.fc2 = nn.Linear(output_size,output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        #print("este es el batch_size: ",x.size(0))
        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        #out, hidden = self.rnn(x, hidden)
        #print("dimensiones del tensor que entra a la rnn: ",x.shape)
        out,_ = self.rnn(x)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        #print("shape de out q entra a la fully connected ",out.shape)
        #import ipdb;ipdb.set_trace()
        print("dimensiones de lo q sale de la rnn: ",out.shape)
        out = self.fc(out.reshape(-1,out.shape[2]*out.shape[1]))
        #out = self.fc2(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

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
        """        
        self.c1 = nn.Conv1d(input_size, hidden_size, 3,padding=1)
        self.a1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(hidden_size)
        self.p1 = nn.AvgPool1d(2)
        
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 3,padding=1)
        self.a2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(hidden_size)
        self.p2 = nn.AvgPool1d(2)
        
        self.c3 = nn.Conv1d(hidden_size, hidden_size, 3,padding=1)
        self.a3 = nn.ReLU()
        self.b3 = nn.BatchNorm1d(hidden_size)
        self.p3 = nn.AvgPool1d(2)
        
        #nn.Sequential(nn.Conv1d(input_size,hidden_size,3),
        #)
        #self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)
        #self.gru = nn.GRU(input_size,hidden_size,n_layers,dropout=0.01)
        self.out = nn.Linear(hidden_size*int(maxlen/2**num_layers), output_size)
        """

    def forward(self, inputs, hidden=None):
        #batch_size = inputs.size(1)
        #batch_size = inputs.size(0)
        #print("dimensiones del tensor que entra a la cnn: ",inputs.shape)
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        #inputs = inputs.transpose(0, 1).transpose(1, 2)

        #entra un tensor de dimensiones: batch x seq_len x input_size
        #quiero un tensor dedimensiones: batch x input_size x seq_len
        inputs = inputs.transpose(1, 2)
        x = self.block(inputs)
        #print(x.shape)
        """
        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        #print("Paso primera capa")
        #usar relu y batch norm a la salida 
        a = self.a1(c)
        b = self.b1(a)
        p = self.p1(b)
        #print("Paso segunda capa")
        c = self.c2(p)
        a = self.a2(c)
        b = self.b2(a)
        p = self.p2(b)
        #print("Paso tercera capa")
        #print("Paso cuarta capa")
        c = self.c3(p)
        a = self.a3(c)
        b = self.b3(a)
        p = self.p3(b)
        #print("dimensiones del tensor que sale de la cnn antes de transpose: ",p.shape)
        
        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        #p = p.transpose(1, 2).transpose(0, 1)
        #print("dimensiones del tensor que entra a la gru antes de transpose: ",p.shape)
        #sale tensor de batch_size x hidden_size x seq_len
        
        #quiero tensor de seq_len x batch_size x hidden_size
        #p = p.transpose(0,1).transpose(0,2)
        #print("dimensiones del tensor que entra a la gru dsp de transpose: ",p.shape)
        #p = torch.tanh(p)
        #output, hidden = self.gru(p, hidden)
        #print("Paso quinta capa")
        #print("dimension de lo que sale de la gru ",output.shape)
        
        #conv_seq_len = p.shape[2]
        #p = p.view(-1,conv_seq_len * self.hidden_size) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        """
        p = torch.flatten(x,start_dim=1)
        #output = torch.tanh(self.out(output))
        #output = self.out(output.reshape(-1,conv_seq_len*self.hidden_size))
        #output = self.out(output.reshape(conv_seq_len*batch_))
        output = self.out(p)
        #print("Paso ultima capa")
        #print("shape de la salida:",output.shape)
        
        #output = output.view(conv_seq_len, -1, self.output_size)
        return output

correctedData = pn.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem.csv",delimiter=',',header=None) #comentar esta linea en caso de descomentar la anterior
cantidad_preg = correctedData.shape[0]
#Xtrain,Ytrain,Xtest,Ytest,Xval,Yval = utils.separate_dataset(correctedData.values,cantidad_preg,True)
print("llego hasta dsp de separate dataset")
#print(correctedData.values[:,1])
text = correctedData.values[:,1]
#labels = correctedData.values[:,0]
#labels = np.array(labels,dtype=np.int8)
# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))
#print(maxlen)
# Padding

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of
# the sentence matches the length of the longest sentence

for i in range(len(text)):
    while(len(text[i]))<maxlen:
        text[i] += ' '

#for i in range(len(Xtrain)):
#    while len(Xtrain[i,0])<maxlen:
#        Xtrain[i,0] += ' ' 

#for i in range(len(Xtest)):
#    while len(Xtest[i,0])<maxlen:
#        Xtest[i,0] += ' '

#for i in range(len(Xval)):
#    while len(Xval[i,0])<maxlen:
#        Xval[i,0] += ' '

print("llego hasta dsp de llenar con whitespace")
#print(len(text[0]))
char = CharTokenizer()
indxChar = np.zeros((len(text),maxlen))
#indxTrain= np.zeros((len(Xtrain),maxlen))
#indxTest = np.zeros((len(Xtest),maxlen))
#indxVal = np.zeros((len(Xval),maxlen))
#indx es una matriz que tiene en cada fila los indices de los caracteres de la oracion, y maxlen columnas
for i in range(len(text)):
    indxChar[i] = char.tokenize(text[i])
#for i in range(len(Xtrain)):
#    indxTrain[i] = char.tokenize(Xtrain[i,0])
#for i in range(len(Xtest)):
#    indxTest[i] = char.tokenize(Xtest[i,0])
#for i in range(len(Xval)):
#    indxVal[i] = char.tokenize(Xval[i,0])
#print(indxTest[0])
#print(indxTest[60])

print("llego hasta dsp de char tokenize")
dict_size = len(char.char_set)
#dict_size = 80
#print(dict_size)
#batch_size_train = len(Xtrain)
#batch_size_test = len(Xtest)
#batch_size_val = len(Xval)
#seq_len = maxlen
cant_preg = len(text)
X = one_hot_encode(indxChar,dict_size,maxlen,cant_preg)
Y = [x[0] for x in correctedData.values]
#input_seq_train = one_hot_encode(indxTrain,dict_size,maxlen,batch_size_train)
#input_seq_test = one_hot_encode(indxTest,dict_size,maxlen,batch_size_test)
#input_seq_val = one_hot_encode(indxVal,dict_size,maxlen,batch_size_val)
#input_seq_train = torch.from_numpy(input_seq_train)
#input_seq_test = torch.from_numpy(input_seq_test)
#input_seq_val = torch.from_numpy(input_seq_val)
#target_seq_train = Ytrain
#target_seq_test = Ytest
#target_seq_val = Yval
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

cantidad_labels = correctedData.values[len(correctedData.values)-1,0] + 1
# Instantiate the model with hyperparameters
#model = sRNN(input_size=dict_size, output_size=cantidad_labels, hidden_dim=12, n_layers=1)
#model = sRNN(input_size=dict_size, output_size=cantidad_labels, hidden_dim=20, n_layers=1,bidirectional=True)
# Define hyperparameters
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
net = NeuralNetClassifier(model,module__input_size = dict_size,module__hidden_size=hidden_size,module__output_size=cantidad_labels,module__maxlen=maxlen,module__num_layers=num_layers,criterion=torch.nn.CrossEntropyLoss,optimizer=torch.optim.Adam,verbose=1,device=device)
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=5)
X_train,X_test,y_train,y_test = train_test_split(X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)
gs.fit(X_train,y_train_tensor)

hidden_size = []
max_epochs = []
batch_size = []
score = []

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
            score.append(results['mean_test_score'][candidate])

report(gs.cv_results_,16)
ejex = batch_size
ejey = max_epochs
print(ejex)
print(ejey)
print(len(ejex))
print(len(ejey))
ejez = [0.5,0.3,0.4,0.33,0.6,0.45,0.75,0.8,0.2,0.47,0.56,0.66,0.9,0.87,0.67,0.43]

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
"""

#Creo el trainset y el testset
trainset = torch.utils.data.TensorDataset(input_seq_train,Ytrain) 
testset = torch.utils.data.TensorDataset(input_seq_test,Ytest)
valset = torch.utils.data.TensorDataset(input_seq_val,Yval)
#Creo el dataloader

cant_test = batch_size_test
cant_val = batch_size_val
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True) 
testloader = torch.utils.data.DataLoader(testset,batch_size=cant_test,shuffle=True)
valloader = torch.utils.data.DataLoader(valset,batch_size=cant_val,shuffle=True)
patience=10
early_stopping = EarlyStopping(verbose=True,patience=patience)
train_losses = [] #vector que guarda el loss para cada epoca
val_losses = []
avg_train_losses = []
avg_val_losses = []
ejex = []


# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters())
#target_prob_train = np.zeros((len(Xtrain),cantidad_labels))
#for i in range(len(target_seq_train)):
#    target_prob_train[i,target_seq_train[i]] = 1
#target_prob_test = np.zeros((len(Xtest),cantidad_labels))
#for i in range(len(target_seq_test)):
#    target_prob_test[i,target_seq_test[i]] = 1
# Training Run
input_seq_train = input_seq_train.to(device)
#target_prob_train = torch.from_numpy(target_prob_train)

input_seq_test= input_seq_test.to(device)
#target_prob_test = torch.from_numpy(target_prob_test)
#print(input_seq_train.shape)
"""
"""
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq_train)
    output = output.to(device)
    #target_seq = target_prob_train.to(device)
    target_seq = Ytrain.to(device)
    
    #print("shape de la salida: ",output.shape)
    #print("shape target: ",target_seq.shape)
    #loss = criterion(output, target_seq.view(-1).long())
    loss = criterion(output, target_seq.long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
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

#print("Salida de la red: ",outNet)
#print("La posta: ",YY[i].item())
#acc = 0
print("SHAPE DE LA SALIDA DE LA RED: ",output.shape)
#tens = nn.functional.softmax(output[0,168,:]) 
#print(nn.functional.softmax(output[0,168,:]))
#print(nn.functional.softmax(output[15,168,:]))
#print(nn.functional.softmax(output[40,168,:]))
##print(nn.functional.softmax(output[0,30,:]))
#print(nn.functional.softmax(output[1,150,:]))
#print(nn.functional.softmax(output[50,130,:]))


#clase = torch.max(nn.functional.softmax(output[0,81,:]))[1].item()
#print(clase)

#prob = nn.functional.softmax(output[-1], dim=0).data
#print("SHAPE DE PROB: ",prob.shape)
acc = balanced_accuracy_score(Ytest,output.argmax(dim=1))
#print(acc)

minposs = avg_val_losses.index(min(avg_val_losses))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
print("Tasa de acierto obtenida: ", acc)
plt.plot(ejex,avg_train_losses)
plt.plot(ejex,avg_val_losses)
plt.show()
"""
"""
#acc = 0
for i in range(len(Ytest)):
    # Taking the class with the highest probability score from the output
    clase = output[i,:].argmax()
    #outNet = torch.max(prob, dim=0)[1].item()
    print("clase predicha: ",clase)
    print("clase groundtruth: ", Ytest[i])

#print(acc/len(Ytest)*100)
"""
#IMPLEMENTAR EARLY STOPPING
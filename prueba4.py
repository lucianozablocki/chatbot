import torch
from torch import nn
import numpy as np
import pandas as pn
from char_tokenizer import CharTokenizer
import utils
from sklearn.metrics import balanced_accuracy_score


def one_hot_encode(indx, dict_size, maxlen, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, maxlen, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(maxlen):
            features[i, u, int(indx[i][u])] = 1
    return features

class sRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(sRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)   
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True) 
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*169, output_size)
        #self.fc2 = nn.Linear(output_size,output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        #print("este es el batch_size: ",x.size(0))
        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        #out, hidden = self.rnn(x, hidden)
        out,_ = self.rnn(x)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        #print("shape de out q entra a la fully connected ",out.shape)
        #import ipdb;ipdb.set_trace()
        print(out.shape)
        out = self.fc(out.reshape(-1,out.shape[2]*out.shape[1]))
        #out = self.fc2(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

correctedData = pn.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem.csv",delimiter=',',header=None) #comentar esta linea en caso de descomentar la anterior
cantidad_preg = correctedData.shape[0]
Xtrain,Ytrain,Xtest,Ytest,Xval,Yval = utils.separate_dataset(correctedData.values,cantidad_preg,True)
print("llego hasta dsp de separate dataset")
#print(correctedData.values[:,1])
text = correctedData.values[:,1]
#labels = correctedData.values[:,0]
#labels = np.array(labels,dtype=np.int8)
# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))
print(maxlen)
# Padding

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of
# the sentence matches the length of the longest sentence

for i in range(len(Xtrain)):
    while len(Xtrain[i,0])<maxlen:
        Xtrain[i,0] += ' ' 
print(len(Xtest))
for i in range(len(Xtest)):
    while len(Xtest[i,0])<maxlen:
        Xtest[i,0] += ' '

print("llego hasta dsp de llenar con whitespace")
#print(len(text[0]))
char = CharTokenizer()
indxTrain= np.zeros((len(Xtrain),maxlen))
indxTest = np.zeros((len(Xtest),maxlen))

#indx es una matriz que tiene en cada fila los indices de los caracteres de la oracion, y maxlen columnas
for i in range(len(Xtrain)):
    indxTrain[i] = char.tokenize(Xtrain[i,0])
for i in range(len(Xtest)):
    indxTest[i] = char.tokenize(Xtest[i,0])
#print(indxTest[0])
#print(indxTest[60])

print("llego hasta dsp de char tokenize")
dict_size = len(char.char_set)
#dict_size = 80
#print(dict_size)
batch_size_train = len(Xtrain)
batch_size_test = len(Xtest)
#seq_len = maxlen
input_seq_train = one_hot_encode(indxTrain,dict_size,maxlen,batch_size_train)
input_seq_test = one_hot_encode(indxTest,dict_size,maxlen,batch_size_test)
input_seq_train = torch.from_numpy(input_seq_train)
input_seq_test = torch.from_numpy(input_seq_test)
target_seq_train = Ytrain
target_seq_test = Ytest
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
model = sRNN(input_size=dict_size, output_size=cantidad_labels, hidden_dim=12, n_layers=1,bidirectional=True)

# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
target_prob_train = np.zeros((len(Xtrain),cantidad_labels))
for i in range(len(target_seq_train)):
    target_prob_train[i,target_seq_train[i]] = 1
target_prob_test = np.zeros((len(Xtest),cantidad_labels))
for i in range(len(target_seq_test)):
    target_prob_test[i,target_seq_test[i]] = 1
# Training Run
input_seq_train = input_seq_train.to(device)
target_prob_train = torch.from_numpy(target_prob_train)

input_seq_test= input_seq_test.to(device)
target_prob_test = torch.from_numpy(target_prob_test)
print(input_seq_train.shape)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq_train)
    output = output.to(device)
    #target_seq = target_prob_train.to(device)
    target_seq = Ytrain.to(device)
    
    print("shape de la salida: ",output.shape)
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
print(acc)
"""
acc = 0
for i in range(len(Ytest)):
    # Taking the class with the highest probability score from the output
    clase = torch.argmax(output[i,:])
    #outNet = torch.max(prob, dim=0)[1].item()

    if clase == Ytest[i]:
        acc+=1

print(acc/len(Ytest)*100)
"""
#IMPLEMENTAR EARLY STOPPING
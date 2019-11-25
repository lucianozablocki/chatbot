import numpy as np
from nltk import word_tokenize
import torch 


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def contarClase(dataset,clase): #metodo que cuenta la cantidad de patrones en 'dataset' que pertenecen a la clase 'clase'
    contador = 0
    for i in range(dataset.shape[0]):
        if dataset[i,0]==clase:
            contador+=1
    return contador

def create_tensor_prom_and_embedding(Xtext,cant_patrones,wordvectors):
    token_idx = 0
    wordSet = {} #diccionario de tokens. clave -> token valor -> indice de la posicion en el tensor de embeddings
    embeddingMat = [] #array (que luego sera convertido a tensor) de embeddings vocabulario propio
    promMat = np.zeros((300,cant_patrones),dtype=np.float32) #array de promedios de vectores de embeddings vocabulario original -> decision: no promediar embeddings no conocidos
    for i in range(cant_patrones):
        words = word_tokenize(Xtext[i,0])
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

def separate_dataset(correctedData,cantidad_preg,validation=True):
    #Separo el dataset en test y train -> un patron de cada clase al subconjunto de test, el resto a train
    clase_actual = -1
    indxTest = [] #lista de indices de preguntas en subconjunto de test
    indxTrain= [] #lista de indices de preguntas en subconjunto de train
    indxVal = []
    Xval = None
    Yval = None
    for i in range(cantidad_preg):
        if correctedData[i,0]!=clase_actual: #cambio de clase
            clase_actual = correctedData[i,0]
            cantidadPregClase = contarClase(correctedData,clase_actual) #cuento cuantas preguntas hay pertenecientes a la clase actual
            if clase_actual!=46 and clase_actual !=103 and clase_actual!=104 and clase_actual !=105: #clases con solo 1 patron no se incluyen en test
                indxTest.append(np.random.randint(0, cantidadPregClase-1)+i) #tomo un indice al azar por clase y lo agrego al subconjunto de test
                if validation:
                    indxVal.append(np.random.randint(0, cantidadPregClase-1)+i)
                #NO ESTA CONTROLADA LA REPETICION DE INDICES EN TEST Y VAL, ATM SE PUEDEN REPETIR
    #Obtengo los indices de train
    indices = range(cantidad_preg)
    indxTrain = list(filter(lambda x: x not in indxTest and x not in indxVal, indices)) #al no meter clases con un solo patron al indxTest, pasan automaticamente al indxTrain
    
    cant_test = len(indxTest)
    cant_train = len(indxTrain)
    if validation: 
        cant_val = len(indxVal)
        Yval = np.zeros((cant_val), dtype=np.int64) #clases de los patrones que estan en el subconjunto de validacion
        Xval = np.zeros((cant_val, 1),  dtype=object)
    
    Ytrain = np.zeros((cant_train),dtype=np.int64) #clases de los patrones que estan en el subconjunto de train
    Xtrain = np.zeros((cant_train, 1),  dtype=object) #opcion p mejorar performance, inicializar como sparse cada matriz X
    Ytest = np.zeros((cant_test), dtype=np.int64) #clases de los patrones que estan en el subconjunto de test
    Xtest = np.zeros((cant_test, 1),  dtype=object)

    contTrain = 0
    contTest = 0
    contVal = 0
    for i in range(cantidad_preg):
        if i in indxTrain:
            Ytrain[contTrain] = correctedData[i,0]
            Xtrain[contTrain] = correctedData[i,1]
            contTrain+=1
        elif i in indxTest:
            Ytest[contTest] = correctedData[i,0]
            Xtest[contTest,:] = correctedData[i,1]
            contTest+=1
        if i in indxVal and validation:
            Yval[contVal] = correctedData[i,0]
            Xval[contVal,:] = correctedData[i,1]
            contVal+=1

    Ytrain = torch.from_numpy(Ytrain)
    print("shape del tensor Ytrain : ",Ytrain.shape)
    Ytest = torch.from_numpy(Ytest)
    print("shape del tensor Ytest : ",Ytest.shape)
    if validation:
        Yval = torch.from_numpy(Yval)
        print("shape del tensor Yval : ",Yval.shape)
        #Xval = torch.from_numpy(Xval)
        #Xval = Xval.float()

    #Xtrain = torch.from_numpy(Xtrain)
    #Xtest = torch.from_numpy(Xtest)
    #Xtrain = Xtrain.float()
    #Xtest = Xtest.float()
    
    return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bow import BOW
from nltk.corpus import stopwords
import math
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import torch
from sklearn.metrics import balanced_accuracy_score
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC,SVC

correctedData = pd.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv",delimiter=',') #comentar esta linea en caso de descomentar la anterior
print("Shape de corrected data: ",correctedData.shape)
cantidad_labels = correctedData.values[len(correctedData.values)-1,0] + 1
cantidad_preg = correctedData.shape[0]
print("Cantidad de clases: ",cantidad_labels)
print("Cantidad de patrones: ",cantidad_preg)
correctedData = correctedData.values

stoplist = stopwords.words('spanish')
text = correctedData[:,1]
bow_unigram = BOW(text,'ascii',stoplist,weighting=True)

Y = np.zeros((cantidad_preg),dtype=np.int64)
for i in range(cantidad_preg):
    Y[i] = correctedData[i,0]
Y = torch.from_numpy(Y)

parameters = {
    'C': [0.1, 1, 10, 100, 1000],  
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    'kernel': ['rbf','linear'],
    'max_iter': [1,2,3,4,5]
}
#net = LinearSVC()
net = SVC()
candidatos = 10
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=candidatos)
X_train,X_test,y_train,y_test = train_test_split(bow_unigram.X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
gs.fit(X_train,y_train)

C = []
gamma = []
kernel = []
max_epochs = []
score = []
std=[]

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
            C.append(results['params'][candidate]['C'])
            gamma.append(results['params'][candidate]['gamma'])
            kernel.append(results['params'][candidate]['kernel'])
            max_epochs.append(results['params'][candidate]['max_iter'])

report(gs.cv_results_,candidatos)  

ejex=C
ejey=gamma
ejez=score

plotx,ploty, = np.meshgrid(np.linspace(np.min(ejex),np.max(ejex),10),\
                           np.linspace(np.min(ejey),np.max(ejey),10))
plotz = interp.griddata((ejex,ejey),ejez,(plotx,ploty),method='linear')
#print(plotz.shape)
#print(plotz)
#for i in range(plotz.shape[0]):
#    print(plotz[i,:]) 
#    print(plotz[~np.isnan(plotz)])
#print(index)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')

#print(X_test.shape)            
predict = gs.best_estimator_.predict(X_test)
#print(probs.shape)

# get training and validation loss
#epochs = [i for i in range(len(gs.best_estimator_.history))]
epochs = [i for i in range(max_epochs[0])]
#train_loss = gs.best_estimator_.history[:,'train_loss']
#train_loss = gs.best_estimator_.loss_curve_

#valid_loss = gs.best_estimator_.history[:,'valid_loss']
acc = balanced_accuracy_score(y_test_tensor,predict)
print("tasa de acierto obtenida: ",acc)
#fig1 = plt.figure()
#plt.plot(epochs,train_loss,'g-')
#plt.plot(epochs,valid_loss,'r-')
#plt.title('Training Loss Curves')
#plt.xlabel('Epochs')
#plt.ylabel('Cross Entropy Loss')
#plt.legend(['Train','Validation'])

plt.show()

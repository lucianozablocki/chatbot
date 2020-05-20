import pandas as pn
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
from sklearn.ensemble import GradientBoostingClassifier


correctedData = pn.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv",delimiter=',',header=None) #comentar esta linea en caso de descomentar la anterior
cantidad_preg = correctedData.shape[0]
correctedData = correctedData.values
stoplist = stopwords.words('spanish')
bow_unigram = BOW(correctedData[:,1],'ascii',stoplist,weighting = False)

Y = np.zeros((cantidad_preg),dtype=np.int64)
for i in range(cantidad_preg):
    Y[i] = correctedData[i,0]

Y = torch.from_numpy(Y)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

parameters = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'random_state': [12]
}

net = GradientBoostingClassifier()

candidatos = 1
gs = RandomizedSearchCV(net,parameters,verbose=2,n_jobs=-2,cv=2,scoring='balanced_accuracy',n_iter=candidatos)
X_train,X_test,y_train,y_test = train_test_split(bow_unigram.X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)

gs.fit(X_train,y_train)

# Number of trees in random forest
n_estimators = []
# Number of features to consider at every split
max_features = []
# Maximum number of levels in tree
max_depth= []
# Minimum number of samples required to split a node
min_samples_split = []
# Minimum number of samples required at each leaf node
min_samples_leaf = []
score=[]
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
            n_estimators.append(results['params'][candidate]['n_estimators'])
            max_features.append(results['params'][candidate]['max_features'])
            max_depth.append(results['params'][candidate]['max_depth'])
            min_samples_split.append(results['params'][candidate]['min_samples_split'])
            min_samples_leaf.append(results['params'][candidate]['min_samples_leaf'])

report(gs.cv_results_,candidatos)  

probs = gs.best_estimator_.predict_proba(X_test)
acc = balanced_accuracy_score(y_test_tensor,np.argmax(probs,axis=1))
print("tasa de acierto obtenida: ",acc)

ejex = n_estimators
ejey = max_depth
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

plt.show()

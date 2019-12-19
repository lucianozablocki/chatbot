# Import the model we are using
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
import utils
from bow import BOW
from nltk.corpus import stopwords
import math
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import torch
from sklearn.metrics import balanced_accuracy_score
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
rf = RandomForestClassifier(random_state = 12)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

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
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

correctedData = pn.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem_completadas.csv",delimiter=',') #comentar esta linea en caso de descomentar la anterior
cantidad_preg = correctedData.shape[0]
correctedData = correctedData.values
print(type(correctedData))
print(correctedData.dtype)
#Xtrain_text,trainY_RF,Xtest_text,testY_RF,_,_ = utils.separate_dataset(correctedData,cantidad_preg,validation=False)
# print('Hasta acá todo ok')
# Instantiate model 
stoplist = stopwords.words('spanish')
#print(Xtrain_text.shape)
bow_unigram = BOW(correctedData[:,1],'ascii',stoplist,weighting = False)

Y = np.zeros((cantidad_preg),dtype=np.int64)
for i in range(cantidad_preg):
    Y[i] = correctedData[i,0]

Y = torch.from_numpy(Y)
print(Y)

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
# Method of selecting samples for training each tree
bootstrap = [True, False]
score=[]
std=[]

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

# Use the random grid to search for best hyperparameters
# First create the base model to tune
#rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
X_train,X_test,y_train,y_test = train_test_split(bow_unigram.X,Y,shuffle=True,stratify=Y,test_size=0.1,random_state=12)
candidatos = 2
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = candidatos, cv = 10, verbose=2, random_state=12, n_jobs = -1,scoring='balanced_accuracy')
# Fit the random search model
y_test_tensor = torch.LongTensor(y_test)
#from collections import Counter
#count = Counter(y_train)
#print(sorted(count.items(), key=lambda pair: pair[1], reverse=True))

rf_random.fit(X_train, y_train)

report(rf_random.cv_results_,candidatos) 
probs = rf_random.best_estimator_.predict_proba(X_test)
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

"""
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    cositas = errors / test_labels.shape[0]
    print(cositas.shape)
    mape = 100 * np.mean(cositas,dim = 0, keepdim = False, dtype = np.float64)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy



base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(trainX_RF, trainY_RF)
basePredictions = base_model.predict(testX_RF)
# base_accuracy = evaluate(base_model, testX_RF, testY_RF)
best_random = rf_random.best_estimator_
bestPredictions = best_random.predict(testX_RF)

print("Shape de testY_RF: ",testY_RF.shape)
print("Shape de Base Predictions: ",basePredictions.shape)
print("Shape de Best Predictions: ",bestPredictions.shape)
baseacc = balanced_accuracy_score(testY_RF,list(map (lambda x: int(x), basePredictions)))
bestacc = balanced_accuracy_score(testY_RF,list(map (lambda x: int(x), bestPredictions)))

print("Accuracy base: ",baseacc)
print("Best accuracy: ",bestacc)
# random_accuracy = evaluate(best_random, testX_RF, testY_RF)

# print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# Import matplotlib for plotting and use magic command for Jupyter Notebooks


# # # Set the style
# # plt.style.use('fivethirtyeight

# print('Instancié el modelo...')
# # rf = RandomForestRegressor(n_estimators= 10000, random_state=42)

# rf_new = RandomForestRegressor(n_estimators = 9000, criterion = 'mse', max_depth = None, 
#                                min_samples_split = 3, min_samples_leaf = 1,verbose=1,n_jobs=-2)

# print('Arranqué a entrenar...')
# # Train the model on training data
# rf_new.fit(trainX_RF, np.ravel(trainY_RF))  

# print('Listo!')
# print('Probando el modelo...')
# # Use the forest's predict method on the test data
# predictions = rf_new.predict(testX_RF)
# print(predictions)
# # Calculate the absolute errors
# errors = abs(predictions - np.ravel(testY_RF))

# # Print out the mean absolute error (mae)
# #print('Average model error:', round(np.mean(errors), 2), 'degrees.')

# # Compare to baseline
# #improvement_baseline = 100 * abs(np.mean(errors) - np.mean(baseline_errors)) / np.mean(baseline_errors)
# #print('Improvement over baseline:', round(improvement_baseline, 2), '%.')

# # Calculate mean absolute percentage error (MAPE)
# # mape = 100 * (errors / 103)
# acc = 0
# for i in range(0,102):
#     if (math.floor(predictions[i])==testY_RF[i]):
#         acc +=1
#     #print("Predije: ",predictions[i]," y en verdad era: ",testY_RF[i])
# print(acc/102*100)
# # Calculate and display accuracy
# #accuracy = 100 - np.mean(mape)
# #print('Accuracy:', round(accuracy, 2), '%.')

"""
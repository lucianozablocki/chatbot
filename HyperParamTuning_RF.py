# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
import utils
from bow import BOW
from nltk.corpus import stopwords
import math
from sklearn.model_selection import RandomizedSearchCV
import torch
from sklearn.metrics import balanced_accuracy_score
rf = RandomForestRegressor(random_state = 42)
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


#trainX_RF = pn.read_csv("trainX_RF.csv",header=None,delimiter=',').values
#trainY_RF = pn.read_csv("trainY_RF.csv",header=None,delimiter=',').values
#testX_RF = pn.read_csv("testX_RF.csv",header=None,delimiter=',').values
#testY_RF = pn.read_csv("testY_RF.csv",header=None,delimiter=',').values



correctedData = pn.read_csv("C:/Users/Juani/chatbot/preprocessedQuestions_lem.csv",delimiter=',') #comentar esta linea en caso de descomentar la anterior
cantidad_preg = correctedData.shape[0]
correctedData = correctedData.values
print(type(correctedData))
print(correctedData.dtype)
Xtrain_text,trainY_RF,Xtest_text,testY_RF,_,_ = utils.separate_dataset(correctedData,cantidad_preg,validation=False)
# print('Hasta acá todo ok')
# Instantiate model 
stoplist = stopwords.words('spanish')
print(Xtrain_text.shape)
bow_unigram = BOW(Xtrain_text.ravel(),'ascii',stoplist,weighting = False)
trainX_RF = bow_unigram.X
testX_RF = bow_unigram.vectorizer.transform(Xtest_text.ravel())
print('Training Features Shape:', trainX_RF.shape)#num_patrones x num_caracteristicas
print('Training Labels Shape:', trainY_RF.shape)#vector columna
print('Testing Features Shape:', testX_RF.shape) 
print('Testing Labels Shape:', testY_RF.shape)#vector columna

trainX_RF = trainX_RF.toarray()
# trainX_RF = torch.from_numpy(trainX_RF)

trainY_RF = trainY_RF.numpy()
# trainY_RF = torch.from_numpy(trainY_RF)

testX_RF = testX_RF.toarray()
# testX_RF = torch.from_numpy(testX_RF)

testY_RF = testY_RF.type(torch.DoubleTensor)
# testY_RF = testY_RF.toarray()
# testY_RF = torch.from_numpy(testY_RF)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(trainX_RF, trainY_RF)

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
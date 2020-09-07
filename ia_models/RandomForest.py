# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
import utils
from bow import BOW
from nltk.corpus import stopwords
import math
#trainX_RF = pn.read_csv("trainX_RF.csv",header=None,delimiter=',').values
#trainY_RF = pn.read_csv("trainY_RF.csv",header=None,delimiter=',').values
#testX_RF = pn.read_csv("testX_RF.csv",header=None,delimiter=',').values
#testY_RF = pn.read_csv("testY_RF.csv",header=None,delimiter=',').values

correctedData = pn.read_csv("C:/Users/lucy/chatbot/preprocessedQuestions_lem.csv",delimiter=',') #comentar esta linea en caso de descomentar la anterior
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

# Import matplotlib for plotting and use magic command for Jupyter Notebooks


# # Set the style
# plt.style.use('fivethirtyeight

print('Instancié el modelo...')
# rf = RandomForestRegressor(n_estimators= 10000, random_state=42)

rf_new = RandomForestRegressor(n_estimators = 9000, criterion = 'mse', max_depth = None, 
                               min_samples_split = 3, min_samples_leaf = 1,verbose=1,n_jobs=-2)

print('Arranqué a entrenar...')
# Train the model on training data
rf_new.fit(trainX_RF, np.ravel(trainY_RF))  

print('Listo!')
print('Probando el modelo...')
# Use the forest's predict method on the test data
predictions = rf_new.predict(testX_RF)
print(predictions)
# Calculate the absolute errors
errors = abs(predictions - np.ravel(testY_RF))

# Print out the mean absolute error (mae)
#print('Average model error:', round(np.mean(errors), 2), 'degrees.')

# Compare to baseline
#improvement_baseline = 100 * abs(np.mean(errors) - np.mean(baseline_errors)) / np.mean(baseline_errors)
#print('Improvement over baseline:', round(improvement_baseline, 2), '%.')

# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / 103)
acc = 0
for i in range(0,102):
    if (math.floor(predictions[i])==testY_RF[i]):
        acc +=1
    #print("Predije: ",predictions[i]," y en verdad era: ",testY_RF[i])
print(acc/102*100)
# Calculate and display accuracy
#accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')
import numpy as np
import pandas as pn
import sys
#nodosAcad = range()
n = 116 #n -> cantidad de respuestas
adjMat = np.zeros((n,n))
resp = pn.read_csv('./csv_files/respuestas - time.csv',header=None)
respMat = resp.values
#print(respMat)
cantidadRespuestas = respMat.shape[0]
# print("Cantidad respuestas:",cantidadRespuestas)
academicas = []
administrativas = []
saludo = []
nodoChat = []
# print(respMat)
for i in range(cantidadRespuestas):
    if respMat[i,2] == 1: #Tipo administrativa
        administrativas.append(respMat[i,0])
        
    elif respMat[i,2] == 2: #Tipo academica
        academicas.append(respMat[i,0])

    elif respMat[i,2] == 3:
        nodoSaludo = respMat[i,0]
    
    elif respMat[i,2] == 4:
        nodoBye = respMat[i,0]
    
    elif respMat[i,2] == 5:
        nodoError = respMat[i,0]

    elif respMat[i,2] == 6:
        nodoErrorAdm = respMat[i,0]    

    elif respMat[i,2] == 7:
        nodoErrorAca = respMat[i,0]    

    elif respMat[i,2] == 8:
        nodoChat.append(respMat[i,0])


# print(academicas)
# print(administrativas)
# print(nodoSaludo, nodoBye, nodoError, nodoErrorAdm, nodoErrorAca, nodoChat)

#Empiezo a crear la matriz...
for i in range (len(academicas)):

    adjMat[academicas[i],academicas] = 1
    adjMat[academicas[i],academicas[i]] = 0 #Para que no pueda volver a emitir la misma respuesta
    adjMat[academicas[i],nodoErrorAca] = 1 
    adjMat[academicas[i],nodoBye] = 1
    adjMat[academicas[i],nodoChat] = 1
    adjMat[academicas[i],administrativas] = 1

for i in range (len(administrativas)):

    adjMat[administrativas[i],administrativas] = 1
    adjMat[administrativas[i],administrativas[i]] = 0 #Para que no pueda volver a emitir la misma respuesta
    adjMat[administrativas[i],nodoErrorAdm] = 1
    adjMat[administrativas[i],nodoBye] = 1    
    adjMat[administrativas[i],nodoChat] = 1
    adjMat[administrativas[i],academicas] = 1

# print(adjMat[11,:])
# print(adjMat[11,:])
# print(adjMat[102,:])

#Flujo del saludo
adjMat[nodoSaludo,nodoError] = 1
adjMat[nodoSaludo,nodoChat[:]] = 1
adjMat[nodoSaludo,academicas] = 1
adjMat[nodoSaludo,administrativas] = 1
adjMat[nodoSaludo,nodoBye] = 1

#Flujo del error
adjMat[nodoError,academicas] = 1
adjMat[nodoError,administrativas] = 1
adjMat[nodoError,nodoChat[:]] = 1
adjMat[nodoError,nodoError] = 1
adjMat[nodoError,nodoBye] = 1

#Flujo de despedirse: se deja todo 0

#Flujo de las interacciones desestructuradas
for i in range(len(nodoChat)):
    adjMat[nodoChat[i],nodoChat] = 1
    adjMat[nodoChat[i],nodoError] = 1
    adjMat[nodoChat[i],nodoBye] = 1
    adjMat[nodoChat[i],academicas] = 1
    adjMat[nodoChat[i],administrativas] = 1

#Flujo del error Academico
adjMat[nodoErrorAca,nodoErrorAca] = 1
adjMat[nodoErrorAca,academicas] = 1

#Flujo del error Administrativo
adjMat[nodoErrorAdm,nodoErrorAdm] = 1
adjMat[nodoErrorAdm,administrativas] = 1

pn.DataFrame(adjMat).to_csv("./adjMat_time.csv", header=None, index=None)

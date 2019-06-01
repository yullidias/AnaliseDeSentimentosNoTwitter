import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix

import analise

#Preprocessing
vocabulary, X_train, y_train, X_test, y_test = analise.inicia()

print("Vocabulario: "+str(len(vocabulary)))
print("Entradas Treino: "+str(len(y_train)))
print("Entradas Teste: "+str(len(y_test)))
print("Entradas Totais: "+str(len(y_train)+len(y_test)))

#Training and Predictions 
classifier = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree')  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

#Evaluating the Algorithm
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
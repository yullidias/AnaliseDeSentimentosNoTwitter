import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix

import analise

#Preprocessing
vocabulary, X_train, y_train, X_test, y_test = analise.inicia()

#Assign colum names to the dataset
names = list(vocabulary.keys())
names.append("Class")
print(len(names))
#Training and Predictions 
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

#Evaluating the Algorithm
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
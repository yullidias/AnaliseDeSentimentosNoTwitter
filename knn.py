from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix

import preprocess

#Preprocessing
X_train, y_train, X_test, y_test = preprocess.preprocess()

print("Entradas Treino: "+str(len(y_train)))
print("Entradas Teste: "+str(len(y_test)))
print("Entradas Totais: "+str(len(y_train)+len(y_test)))

#Training and Predictions 
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train, y_train['Polaridade']) 
y_pred = classifier.predict(X_test)

#Evaluating the Algorithm
print(confusion_matrix(y_test['Polaridade'], y_pred))  
print(classification_report(y_test['Polaridade'], y_pred)) 
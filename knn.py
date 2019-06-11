from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import preprocess
import sys
import numpy as np
import matplotlib.pyplot as plt


N = 11
def findBestK():
    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=False)
    accuracity = []
    for x in range(1,N):
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(list(range(1,N)),accuracity,color='green')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=True)
    accuracity = []
    for x in range(1,N):
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(list(range(1,N)),accuracity,color='red')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False)
    accuracity = []
    for x in range(1,N):
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(list(range(1,N)),accuracity,color='blue')
    plt.show()

accuracity = []
DF = [0.2,0.4,0.6,0.8]
def findBestDF():
    for x in DF:
        X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False,min_df=x)
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=4))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
        print(str(x)+" - "+str(w))
    plt.plot(DF,accuracity,color='black')
    plt.show()

X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False,min_df=0.2)
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=4))
classifier.fit(X_train, y_train['Polaridade']) 
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test['Polaridade'], y_pred))  
print(classification_report(y_test['Polaridade'], y_pred))
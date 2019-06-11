from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import preprocess
import sys
import numpy as np
import matplotlib.pyplot as plt


myRange =[0.00001,0.00005,0.0001,0.0005,0.001]

def findBestC(): #encontrar o melhor valor para regularização#
    plt.title("Regressão Logística")
    plt.xlabel("Fator de Regularização")
    plt.ylabel("Acurácia")

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=False)
    accuracity = []
    for x in myRange:
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(myRange, accuracity, color='green', label='Nenhum')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=True)
    accuracity = []
    for x in myRange:
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(myRange,accuracity,color='red', label='Stem Words')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False)
    accuracity = []
    for x in myRange:
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(myRange,accuracity,color='blue', label='Stop Words')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=True)
    accuracity = []
    for x in myRange:
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(myRange,accuracity,color='yellow', label='Ambos')    
    plt.legend(loc='lower right')
    plt.show()


accuracity = []
DF = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
def findBestDF():
    plt.title("Regressão Logística")
    plt.xlabel("MIN_DF")
    plt.ylabel("Acurácia")
    for x in DF:
        X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=False, min_df=x)
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=0.00001))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
        print(str(x)+" - "+str(w))
    plt.plot(DF,accuracity,color='black')
    plt.show()


X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=False, min_df=0.2)
classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=0.00001))
classifier.fit(X_train, y_train['Polaridade']) 
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test['Polaridade'], y_pred))  
print(classification_report(y_test['Polaridade'], y_pred))

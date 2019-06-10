from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import preprocess
import sys
import numpy as np
import matplotlib.pyplot as plt


myRange = [0.001,0.01,0.1,1,10]

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
    plt.plot(myRange, accuracity, color='green', label='None')

    X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=False, isToStemWords=True)
    accuracity = []
    for x in myRange:
        classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear',C=x))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
    plt.plot(myRange,accuracity,color='red', label='Stem')

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
    plt.plot(myRange,accuracity,color='yellow', label='Both')

    plt.show()


accuracity = []
DF = [0.2,0.4,0.6,0.8]
def findBestDF():
    for x in DF:
        X_train, y_train, X_test, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False, min_df=x)
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=4))
        classifier.fit(X_train, y_train['Polaridade']) 
        y_pred = classifier.predict(X_test)
        w = accuracy_score(y_pred,y_test['Polaridade'])
        accuracity.append(w)
        print(str(x)+" - "+str(w))
    plt.plot(DF,accuracity,color='black')
    plt.show()



findBestC()
findBestDF()

'''
X_train, X_test, y_train, y_test = preprocess.preprocess(isToRemoveStopWords=True, isToStemWords=False,min_df=0.2)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=4))
classifier.fit(X_train, y_train['Polaridade']) 
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test['Polaridade'], y_pred))  
print(classification_report(y_test['Polaridade'], y_pred))
'''
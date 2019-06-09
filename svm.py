import preprocess
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

def SVM(kernel, isToStemWords, isToRemoveStopWords):
    train = []
    microAvg = []
    x = []
    print("kernel=", kernel, "isToStemWords=", isToStemWords, "isToRemoveStopWords=",isToRemoveStopWords, "\n")

    vocabularioTreino, training, vocabularioTest, test = preprocess.preprocess(isToStemWords=isToStemWords, isToRemoveStopWords=isToRemoveStopWords)

    classifier = svm.SVC(kernel=kernel, gamma='auto', decision_function_shape="ovo")

    classifier.fit(vocabularioTreino, training['Polaridade'])
    predictTrain = classifier.predict(vocabularioTreino)
    prediction = classifier.predict(vocabularioTest)

    print(classification_report(test['Polaridade'], prediction)) #, output_dict=True))
    print(confusion_matrix(test['Polaridade'], prediction))
    print(" --- END ---")

SVM('linear', True, True)
SVM('poly', True, True)
SVM('rbf', True, True)
SVM('sigmoid', True, True)

SVM('linear', True, False)
SVM('poly', True, False)
SVM('rbf', True, False)
SVM('sigmoid', True, False)

SVM('linear', False, True)
SVM('poly', False, True)
SVM('rbf', False, True)
SVM('sigmoid', False, True)

SVM('linear', False, False)
SVM('poly', False, False)
SVM('rbf', False, False)
SVM('sigmoid', False, False)

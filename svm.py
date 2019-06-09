import preprocess
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

vocabularioTreino, training, vocabularioTest, test = preprocess.preprocess(isToRemoveStopWords=True)

print(len(training.loc[training['Polaridade'] == 1 ]))
print(len(training.loc[training['Polaridade'] == -1 ]))
print(len(training.loc[training['Polaridade'] == 0 ]))

classifier = svm.SVC(kernel='linear', decision_function_shape="ovo")
classifier.fit(vocabularioTreino, training['Polaridade'])
prediction = classifier.predict(vocabularioTest)

print(classification_report(test['Polaridade'], prediction)) #, output_dict=True))
print(confusion_matrix(test['Polaridade'], prediction))

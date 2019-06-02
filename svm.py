import preprocess
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


vocabularioTreino, training, vocabularioTest, test = preprocess.preprocess()

classifier = svm.SVC(kernel='poly', gamma='auto')
classifier.fit(vocabularioTreino, training['Polaridade'])
prediction = classifier.predict(vocabularioTest)

print(classification_report(test['Polaridade'], prediction)) #, output_dict=True))
# print(confusion_matrix(test['Polaridade'], prediction))

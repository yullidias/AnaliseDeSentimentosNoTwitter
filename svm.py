import preprocess
from sklearn import svm
from sklearn.metrics import classification_report

vocabularioTreino, training, vocabularioTest, test = preprocess.preprocess()

classifier = svm.SVC(kernel='poly')
classifier.fit(vocabularioTreino, training['Polaridade'])
prediction = classifier.predict(vocabularioTest)

report = classification_report(test['Polaridade'], prediction)#, output_dict=True)
print(report)

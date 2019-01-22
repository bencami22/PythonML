# Load libraries to be used
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

print('Starting PythonMl')

names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
dataset = pandas.read_csv('Data/winequality-red.csv', names=names)

# Split-out validation dataset
#We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
array = dataset.values
X = array[:,0:12]
Y = array[:,11]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#X_train = X_train.astype('float')
#Y_train = Y_train.astype('float')
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier())) #0.599692 (0.022750)
models.append(('CART', DecisionTreeClassifier())) #1.000000 (0.000000)
models.append(('Naive Bayes', GaussianNB())) # 1.000000 (0.000000)
models.append(('Neural network',MLPClassifier())) #0.924182 (0.024175)
models.append(('Neural network',MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))) #0.526169 (0.111367)

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(f'knn accuracy: {accuracy_score(Y_validation, predictions)}')
print(f'knn confusion matrix: {confusion_matrix(Y_validation, predictions)}')
print(f'knn classification report: {classification_report(Y_validation, predictions)}')

nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print(f'nb accuracy: {accuracy_score(Y_validation, predictions)}')
print(f'nb confusion matrix: {confusion_matrix(Y_validation, predictions)}')
print(f'nb classification report: {classification_report(Y_validation, predictions)}')
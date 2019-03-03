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

X = dataset.drop('quality', axis=1)
Y = dataset['quality']
validation_size = 0.2
seed = 1
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric - see options https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier())) #0.507450 std: 0.030507
models.append(('CART', DecisionTreeClassifier())) #0.611491 std: 0.044385
models.append(('Naive Bayes', GaussianNB())) # 0.532480 std: 0.040657
models.append(('Neural network', MLPClassifier(max_iter=400))) #0.576224 std: 0.034451
models.append(('Neural network',MLPClassifier(max_iter=400, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=seed))) #0.496444 std: 0.061359

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: mean: %f std: %f" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Make predictions on validation dataset
for name, model in models:
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	print(f'{name} accuracy: {accuracy_score(Y_validation, predictions)}')
	print(f'{name} confusion matrix: {confusion_matrix(Y_validation, predictions)}')
	print(f'{name} classification report: {classification_report(Y_validation, predictions)}')
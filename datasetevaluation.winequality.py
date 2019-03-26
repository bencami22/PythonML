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
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

print('Starting PythonMl')

names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
dataset = pandas.read_csv('Data/winequality-red.csv', names=names)

#bins/buckets
dataset['volatile acidity']=pandas.cut(dataset['volatile acidity'], [0, 0.3, 0.6, 0.9, 1.2, 1.6], labels=[1,2,3,4,5])
dataset['fixed acidity']=pandas.cut(dataset['fixed acidity'], [0, 6, 8, 10, 12, 14, 16], labels=[1,2,3,4,5,6])
dataset['residual sugar']=pandas.cut(dataset['residual sugar'], [0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20], labels=[1,2,3,4,5,6,7,8,9,10])
dataset['chlorides']=pandas.cut(dataset['chlorides'], [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.7], labels=[1,2,3,4,5,6,7,8])
#dataset['citric acid']=pandas.cut(dataset['citric acid'], [0, 0.02, 0.04, 0.06, 0.08, 0.1], labels=[1, 2, 3, 4, 5])
dataset['free sulfur dioxide']=pandas.cut(dataset['free sulfur dioxide'], [0, 10, 20, 30, 40, 50, 60, 70, 80], labels=[1, 2, 3, 4, 5, 6, 7, 8])
dataset['total sulfur dioxide']=pandas.cut(dataset['total sulfur dioxide'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['sulphates']=pandas.cut(dataset['sulphates'], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1], labels=[1, 2, 3, 4, 5, 6, 7])
dataset['alcohol']=pandas.cut(dataset['alcohol'], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
dataset['density']=pandas.cut(dataset['density'], [0, 0.990, 0.995, 0.996, 0.997, 0.999, 2], labels=[1, 2, 3, 4, 5, 6])
dataset['pH']=pandas.cut(dataset['pH'], [0, 3, 3.2, 3.4, 3.6, 3.8, 4, 5], labels=[1, 2, 3, 4, 5, 6, 7])
# Split-out validation dataset
#We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.

X = dataset.drop('quality', axis=1)
Y = dataset['quality']
validation_size = 0.2
seed = 1
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print(f'X_train: {X_train.shape}')
print(f'X_validation: {X_validation.shape}')
print(f'Y_train: {Y_train.shape}')
print(f'Y_validation: {Y_validation.shape}')
# Test options and evaluation metric - see options https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier(3))) #0.507450 std: 0.030507
models.append(('SVC', SVC(kernel="linear", C=0.025))) #0.611491 std: 0.044385
models.append(('SVC2', SVC(gamma=2, C=1))) #0.611491 std: 0.044385
models.append(('CART', DecisionTreeClassifier(max_depth=5))) #0.611491 std: 0.044385
#models.append(('GaussianProcessClassifier', GaussianProcessClassifier(1.0 * RBF(1.0)))) # 0.532480 std: 0.040657
models.append(('RandomForestClassifier', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))) # 0.532480 std: 0.040657
models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis())) # 0.532480 std: 0.040657
models.append(('AdaBoostClassifier', AdaBoostClassifier())) # 0.532480 std: 0.040657
models.append(('Naive Bayes', GaussianNB())) # 0.532480 std: 0.040657
models.append(('Neural network', MLPClassifier(max_iter=400))) #0.576224 std: 0.034451
models.append(('Neural network',MLPClassifier(max_iter=400, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=seed))) #0.496444 std: 0.061359

# evaluate each model in turn
#Calculate the mean of these measures to get an idea of how well the procedure performs on average.
#Calculate the standard deviation of these measures to get an idea of how much the skill of the procedure is expected to vary in practice.
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
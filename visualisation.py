# Load libraries to be used
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print('Starting PythonMl')

names = ['TypicalOrDegraded','Mass','FallObserved','Year','RecLat','RecLong']
dataset = pandas.read_csv('Data/meteorite-landings-visualisation.csv', names=names)

# box and whisker plots
dataset=dataset.astype(float)

#dataset.plot(kind='box', subplots=True, layout=(2,6), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
# Load libraries
import pandas

print('Starting PythonMl')

names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
dataset = pandas.read_csv('Data/winequality-red.csv', names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('quality').size())

#dataset columns and data types of each column. 
print(dataset.info())
# Load libraries
import pandas

print('Starting PythonMl')

names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
dataset = pandas.read_csv('Data/winequality-red.csv', names=names)

#bins/buckets
dataset['volatile acidity']=pandas.cut(dataset['volatile acidity'], [0, 0.3, 0.6, 0.9, 1.2, 1.6])
dataset['fixed acidity']=pandas.cut(dataset['fixed acidity'], [0, 6, 8, 10, 12, 14, 16])
dataset['residual sugar']=pandas.cut(dataset['residual sugar'], [0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20])
dataset['chlorides']=pandas.cut(dataset['chlorides'], [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.7])
#dataset['citric acid']=pandas.cut(dataset['citric acid'], [0, 0.02, 0.04, 0.06, 0.08, 0.1])
dataset['free sulfur dioxide']=pandas.cut(dataset['free sulfur dioxide'], [0, 10, 20, 30, 40, 50, 60, 70, 80])
dataset['total sulfur dioxide']=pandas.cut(dataset['total sulfur dioxide'], [0, 10, 20, 30, 40, 50, 60, 70, 80,100,200,300])
dataset['sulphates']=pandas.cut(dataset['sulphates'], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
dataset['alcohol']=pandas.cut(dataset['alcohol'], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
dataset['density']=pandas.cut(dataset['density'], [0, 0.990, 0.995, 0.996, 0.997, 0.999, 2])
dataset['pH']=pandas.cut(dataset['pH'], [0, 3, 3.2, 3.4, 3.6, 3.8, 4, 5])

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
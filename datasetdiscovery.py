# Load libraries
import pandas

print('Starting PythonMl')

names = ['TypicalOrDegraded','Mass','FallObserved','Year','RecLat','RecLong','RecClass']
dataset = pandas.read_csv('Data/meteorite-landings-evaluation.csv', names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('RecClass').size())

#dataset columns and data types of each column. 
print(dataset.info())
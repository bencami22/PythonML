import scipy
import numpy
import csv

print('Starting PythonMl')


with open('Data/meteorite-landings.csv', mode='r', encoding="utf8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        print(f'name:{row["name"]}   id:{row["id"]}   nametype:{row["nametype"]}   recclass:{row["recclass"]}   mass:{row["mass"]}   fall:{row["fall"]}   year:{row["year"]}   reclat:{row["reclat"]}   reclong:{row["reclong"]}   GeoLocation:{row["GeoLocation"]}')
        line_count += 1
    print(f'Processed {line_count} lines.')
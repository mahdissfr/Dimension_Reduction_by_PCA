import csv


def read_from_file():
    x = []
    y = []
    with open('Dataset\Dataset1.csv') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        for row in read:
            x.append(row[0])
            y.append(row[1])
        for i in range(1,len(x)):
            x[i]=float(x[i])
            y[i]=float(y[i])
    return x[1:] , y[1:]
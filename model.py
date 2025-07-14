import numpy as np
import csv

with open("dataset.csv", "r") as f:
    reader = csv.reader(f)
    dataset = [ [float(x) for x in row[1:]] for row in reader ]

a = np.load("a.npy") 
b = np.load("b.npy")  

for row in dataset:
    x = np.array(row)
    y = np.sum(x * a + b)
    print(y)
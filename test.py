import csv
import matplotlib.pyplot as plt
import json

with open("params.json", "r") as f:
    params = json.load(f)

bias = float(params["bias"])
weight = float(params["weight"])

with open("dataset.csv", "r") as f:
    reader = csv.reader(f)
    dataset = [[float(x) for x in row] for row in reader]

y = [row[0] for row in dataset]
x = [row[1] for row in dataset]

min_x = min(x)
max_x = max(x)
line_x = [min_x, max_x]
line_y = [weight * x + bias for x in line_x]

plt.scatter(x, y, color="black", marker="x")
plt.plot(line_x, line_y, color="red")
plt.grid()
plt.show()
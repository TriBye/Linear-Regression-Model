import csv
import random
import json

try:
    with open("params.json", "r") as f:
        params = json.load(f)
    bias = float(params["bias"])
    weight = float(params["weight"])
except:
    bias = 0.0
    weight = random.uniform(-0.01, 0.01)

with open("dataset.csv", "r") as f:
    dataset = [[float(x) for x in row] for row in csv.reader(f)]

n = len(dataset)
epochs = 10000
learn_rate = 1e-5

old_mse = None

for epoch in range(epochs):
    error = []
    mse = 0.0

    for row in dataset:
        y = row[0]
        x = row[1]
        y_p = weight * x + bias
        error_1 = y_p - y
        error.append(error_1)
        mse += error_1 ** 2

    mse /= n

    if epoch == 0:
        old_mse = mse

    grad_weight = 0.0
    grad_bias = 0.0

    for i, row in enumerate(dataset):
        x = row[1]
        grad_weight -= 2 * error[i] * x / n
        grad_bias -= 2 * error[i] / n

    weight += learn_rate * grad_weight
    bias += learn_rate * grad_bias

    if epoch % 100 == 0 or epoch == epochs - 1:
        print("--------------------")
        print(f"Epoch: #{epoch}")
        print(f"MSE: {mse}")

print("\n--Training--")
print(f"Old Loss: {old_mse}")
print(f"New Loss: {mse}")
print(f"diff: {old_mse - mse}")

with open("params.json", "w") as f:
    json.dump({"bias": bias, "weight": weight}, f)
import csv
import random
import json
import matplotlib.pyplot as plt

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
initial_learn_rate = 1e-2
learn_rate = initial_learn_rate
epsilon_rel = 1e-6
decay_factor = 0.98
decay_interval = 100

old_mse = None
epoch = 0
mse_list = []

while True:
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
    mse_list.append(mse)

    grad_weight = 0.0
    grad_bias = 0.0

    for i, row in enumerate(dataset):
        x = row[1]
        grad_weight -= 2 * error[i] * x / n
        grad_bias -= 2 * error[i] / n

    weight += learn_rate * grad_weight
    bias += learn_rate * grad_bias

    if epoch % decay_interval == 0 and epoch > 0:
        learn_rate *= decay_factor

    if epoch % 100 == 0:
        print("--------------------")
        print(f"Epoch: #{epoch}")
        print(f"MSE: {mse}")
        print(f"Lernrate: {learn_rate}")

    if old_mse is not None:
        mse_change = abs(old_mse - mse) / abs(old_mse) if old_mse != 0 else 0.0
        if mse_change < epsilon_rel:
            print("Training gestoppt wegen Konvergenzbedingung.")
            break

    old_mse = mse
    epoch += 1

with open("params.json", "w") as f:
    json.dump({"bias": bias, "weight": weight}, f)

plt.plot(mse_list)
plt.xlabel("Epoche")
plt.ylabel("Loss (MSE)")
plt.title("Loss-Kurve während des Trainings")
plt.show()
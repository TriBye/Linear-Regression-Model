import csv
import random
import json
import matplotlib.pyplot as plt

try:
    with open("model_params.json", "r") as f:
        params = json.load(f)
    bias = float(params["bias"])
    weight = float(params["weight"])
except:
    bias = 0.0
    weight = random.uniform(-0.01, 0.01)

with open("dataset.csv", "r") as f:
    dataset = [[float(x) for x in row] for row in csv.reader(f)]

n = len(dataset)
learn_rate = 1e-5
epsilon_rel = 1e-6
max_epochs = 10**6

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

    if epoch % 100 == 0 or epoch == max_epochs - 1:
        print("--------------------")
        print(f"Epoch: #{epoch}")
        print(f"MSE: {mse}")

    if old_mse is not None:
        mse_change = abs(old_mse - mse) / abs(old_mse) if old_mse != 0 else 0.0
        if mse_change < epsilon_rel:
            print("Training gestoppt wegen Konvergenzbedingung.")
            break

    old_mse = mse
    epoch += 1
    if epoch >= max_epochs:
        print("Training gestoppt wegen max_epochs.")
        break

print("\n--Training--")
print(f"Letzte Loss: {mse}")
print(f"Letzte Epoche: {epoch}")

with open("model_params.json", "w") as f:
    json.dump({"bias": bias, "weight": weight}, f)

plt.plot(mse_list)
plt.xlabel("Epoche")
plt.ylabel("Loss (MSE)")
plt.title("Verlauf des Loss während des Trainings")
plt.show()
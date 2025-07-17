import csv
import torch
import matplotlib.pyplot as plt

with open("dataset.csv", "r") as f:
    dataset = [[float(x) for x in row] for row in csv.reader(f)]

x_vals = [row[1] for row in dataset]
y_vals = [row[0] for row in dataset]

X = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

mse_list = []
epsilon_rel = 1e-6
old_loss = None
max_epochs = 100000

for epoch in range(max_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    mse = loss.item()
    mse_list.append(mse)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print("--------------------")
        print(f"Epoch: #{epoch}")
        print(f"MSE: {mse}")
    
    if old_loss is not None:
        mse_change = abs(old_loss - mse) / abs(old_loss) if old_loss != 0 else 0.0
        if mse_change < epsilon_rel:
            print("Training gestoppt wegen Konvergenzbedingung.")
            break
    old_loss = mse

print("\n--Training--")
print(f"Letzte Loss: {mse}")
print(f"Letzte Epoche: {epoch}")
print(f"Gewicht: {model.weight.item()}")
print(f"Bias: {model.bias.item()}")

plt.plot(mse_list)
plt.xlabel("Epoche")
plt.ylabel("Loss (MSE)")
plt.title("Loss-Kurve während des Trainings")
plt.show()
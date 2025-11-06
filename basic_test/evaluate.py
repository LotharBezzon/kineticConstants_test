import torch
from model import EdgePredictorGNN
from prepare_data import train_loader, val_loader

for batch in train_loader:
    print(batch.edge_attr.shape)
    break

best_model = 10  # Change this to load a different epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 2
hidden_channels = 32
edge_out_channels = 1

# Load model
model = EdgePredictorGNN(in_channels, hidden_channels, edge_out_channels).to(device)
model.load_state_dict(torch.load(f"model_epoch_{best_model}.pt", map_location=device))
model.eval()

def predict(loader, name):
	all_preds = []
	all_targets = []
	with torch.no_grad():
		for batch in loader:
			batch = batch.to(device)
			pred = model(batch)
			all_preds.append(pred.cpu())
			all_targets.append(batch.edge_attr.cpu())
	preds = torch.cat(all_preds, dim=0)
	targets = torch.cat(all_targets, dim=0)
	mse = torch.mean((preds - targets) ** 2).item()
	print(f"{name} set: MSE = {mse:.6f}")
	return preds, targets

print("Evaluating model...")
train_preds, train_targets = predict(train_loader, "Train")
val_preds, val_targets = predict(val_loader, "Validation")

import matplotlib.pyplot as plt

plt.scatter(train_targets[:,0], train_preds[:,0], s=0.005)
plt.show()

plt.scatter(val_targets[:,0], val_preds[:,0], s=0.005)
plt.show()

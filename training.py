# Training script for EdgePredictorGNN
import torch
from model import EdgePredictorGNN
from prepare_data import train_loader, val_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 2  # node feature size
hidden_channels = 64
edge_out_channels = 1  # edge feature size
lr = 1e-3
epochs = 10

# Model, optimizer, loss
model = EdgePredictorGNN(in_channels, hidden_channels, edge_out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_fn = torch.nn.MSELoss()

def train():
	model.train()
	total_loss = 0
	for batch in train_loader:
		batch = batch.to(device)
		optimizer.zero_grad()
		pred = model(batch)
		loss = loss_fn(pred, batch.edge_attr)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * batch.num_graphs
	return total_loss / len(train_loader.dataset)

def validate():
	model.eval()
	total_loss = 0
	with torch.no_grad():
		for batch in val_loader:
			batch = batch.to(device)
			pred = model(batch)
			loss = loss_fn(pred, batch.edge_attr)
			total_loss += loss.item() * batch.num_graphs
	return total_loss / len(val_loader.dataset)

for epoch in range(1, epochs + 1):
	train_loss = train()
	val_loss = validate()
	scheduler.step()
	with open("training_log.txt", "a") as f:
		f.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Learning Rate = {scheduler.get_last_lr()[0]:.4f}\n")
	print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
	if epoch % 2 == 0:
		torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
		print(f"Model parameters saved at epoch {epoch}.")

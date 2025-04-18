import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train_model(model, train_dataset, val_dataset, num_epochs=800, lr=0.01, lam=1e-10, patience=50, device='cpu'):
    train_loader  = DataLoader(dataset=train_dataset, batch_size=24, shuffle = False)
    val_loader    = DataLoader(dataset=val_dataset, batch_size=24, shuffle = False)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=lam)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_function = torch.nn.MSELoss()

    train_loss_list = []
    val_loss_list = []
    best_loss = float('inf')
    counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for X in train_loader:
            X = X.to(device)
            reconstructed = model(X)
            loss = loss_function(reconstructed, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for X in val_loader:
                X = X.to(device)
                val_pred = model(X)
                val_loss = loss_function(val_pred, X)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

        train_loss_list.append(loss.item())
        val_loss_list.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Train Loss: {loss.item():.4f} - Val Loss: {avg_val_loss:.4f}")

    print(f"Training complete in {time.time() - start_time:.2f} seconds.")
    return train_loss_list, val_loss_list
"""Forecast training utilities: train loop, early stopping, data splitting."""
import torch


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, patience=10, max_epochs=100):
    """
    Train a forecast model with early stopping.

    Args:
        model        : nn.Module
        train_loader : DataLoader for training set
        val_loader   : DataLoader for validation set
        criterion    : loss function (e.g. nn.MSELoss())
        optimizer    : torch optimiser
        device       : 'cuda' or 'cpu'
        patience     : early stopping patience (epochs without improvement)
        max_epochs   : maximum number of epochs

    Returns:
        model         : trained model (best val checkpoint restored)
        best_val_loss : float
    """
    best_val_loss   = float('inf')
    no_improve      = 0
    best_state_dict = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for enc, dec, tgt in train_loader:
            enc  = enc.to(device)
            dec  = dec.to(device)
            y    = tgt['prices'].to(device)
            pred = model(enc, dec, y)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc = enc.to(device)
                dec = dec.to(device)
                y   = tgt['prices'].to(device)
                val_loss += criterion(model(enc, dec), y).item()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss   = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f'Epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  '
              f'best={best_val_loss:.6f}{"  *" if improved else ""}')

        if no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}.')
            break

    if best_state_dict:
        model.load_state_dict(best_state_dict)
    return model, best_val_loss


def simple_split(data, train_ratio=0.4, val_ratio=0.1):
    """Chronological (non-shuffled) train / val / test split."""
    n         = len(data)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

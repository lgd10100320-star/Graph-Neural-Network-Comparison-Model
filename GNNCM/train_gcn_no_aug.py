import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import pickle
import os
import json
from tqdm import tqdm

from models.gcl_model import GCLModel

CONFIG = {
    "encoder_name": "gcn",
    "run_name": "gcn_no_aug",
    "input_dim": 19,
    "hidden_dim": 256,
    "projection_dim": 128,
    "num_layers": 5,
    "dropout": 0.5,
    "temperature": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "epochs": 100,
    "data_path": r"data\processed\processed_data.pkl",
    "save_dir": r"results\no_aug_checkpoints",
    "log_dir": r"logs\tensorboard"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    config_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['run_name']}_config.json")
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)

    print("Loading preprocessed data...")
    with open(CONFIG["data_path"], 'rb') as f:
        train_data = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    print(f"Data loaded. Number of graphs: {len(train_data)}, Number of batches: {len(train_loader)}")

    print(f"Initializing {CONFIG['encoder_name'].upper()} based GCL model (no augmentation)...")
    model = GCLModel(
        encoder_name=CONFIG["encoder_name"],
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        projection_dim=CONFIG["projection_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        temperature=CONFIG["temperature"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    writer = SummaryWriter(log_dir=os.path.join(CONFIG["log_dir"], CONFIG["run_name"]))
    loss_history = []

    print("Starting training without augmentation...")
    min_loss = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch_data in progress_bar:
            optimizer.zero_grad()

            batch_data = batch_data.to(device)
            view_one = batch_data
            view_two = batch_data.clone()

            p1, p2 = model(view_one, view_two)
            loss = model.contrastive_loss(p1, p2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        loss_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['run_name']}_loss_history.json")
        with open(loss_path, 'w') as lf:
            json.dump(loss_history, lf, indent=4)

        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch}/{CONFIG['epochs']}, Average Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['run_name']}_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with loss {avg_loss:.4f}")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()

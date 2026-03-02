import importlib.util
import json
import os
import pickle
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.gcl_model import GCLModel
from utils.augmentations import Augmentation


def load_augmentation_preset(preset_name: str):
    if not preset_name:
        return None

    utils_dir = Path(__file__).resolve().parent / "utils"
    preset_path = utils_dir / f"{preset_name}.py"
    if not preset_path.exists():
        raise FileNotFoundError(f"Augmentation preset file not found: {preset_path}")

    module_name = preset_name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, str(preset_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load augmentation preset: {preset_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_rates"):
        return module.get_rates()
    if hasattr(module, "AUGMENTATION_RATES"):
        return dict(module.AUGMENTATION_RATES)

    raise AttributeError(
        f"Augmentation preset does not define get_rates or AUGMENTATION_RATES: {preset_path}"
    )


CONFIG = {
    "encoder_name": "gin",
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
    "augmentation_methods": ["node_masking", "edge_deletion", "subgraph_deletion"],
    "augmentation_preset": "augmentations_0.3",
    "augmentation_presets": ["augmentations_0.1", "augmentations_0.2", "augmentations_0.25", "augmentations_0.3"],
    "augmentation_rates": {},
    "data_path": r"data\processed\processed_data.pkl",
    "save_dir": r"results\checkpoints",
    "log_dir": r"logs\tensorboard",
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    preset_name = CONFIG.get("augmentation_preset")
    if preset_name:
        aug_tag = preset_name.replace(".", "_")
    else:
        aug_tag = "aug_default_0_15" if not CONFIG.get("augmentation_rates") else "aug_custom"

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["save_dir"], f"config_gin_{aug_tag}.json"), "w", encoding="utf-8") as handle:
        json.dump(CONFIG, handle, indent=4)

    print("Loading preprocessed data...")
    with open(CONFIG["data_path"], "rb") as handle:
        train_data = pickle.load(handle)

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    print(f"Data loaded. Number of graphs: {len(train_data)}, Number of batches: {len(train_loader)}")

    print(f"Initializing {CONFIG['encoder_name'].upper()} based GCL model...")
    model = GCLModel(
        encoder_name=CONFIG["encoder_name"],
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        projection_dim=CONFIG["projection_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        temperature=CONFIG["temperature"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    preset_rates = load_augmentation_preset(CONFIG.get("augmentation_preset"))
    augmentation_rates = preset_rates if preset_rates is not None else CONFIG["augmentation_rates"]
    augmentation = Augmentation(CONFIG["augmentation_methods"], augmentation_rates)

    writer = SummaryWriter(log_dir=os.path.join(CONFIG["log_dir"], CONFIG["encoder_name"]))
    loss_history = []

    print("Starting training...")
    min_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch_data in progress_bar:
            optimizer.zero_grad()

            data1, data2 = augmentation(batch_data)
            data1 = data1.to(device)
            data2 = data2.to(device)

            proj1, proj2 = model(data1, data2)
            loss = model.contrastive_loss(proj1, proj2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        loss_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['encoder_name']}_{aug_tag}_loss_history.json")
        with open(loss_path, "w", encoding="utf-8") as handle:
            json.dump(loss_history, handle, indent=4)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch}/{CONFIG['epochs']}, Average Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['encoder_name']}_{aug_tag}_best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with loss {avg_loss:.4f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()

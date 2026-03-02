import os
import json
import torch
from torch_geometric.loader import DataLoader

from models.gcl_model import GCLModel
import evaluate as base_eval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NO_AUG_DIR = os.path.join(ROOT_DIR, "results", "no_aug_checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "no_aug_checkpoints")
RUNS = ["gcn_no_aug", "gin_no_aug"]

def _candidate_paths(run_name: str, filename: str) -> list[str]:

    return [
        os.path.join(NO_AUG_DIR, run_name, filename),
        os.path.join(NO_AUG_DIR, f"{run_name}_{filename}"),
    ]

def evaluate_run(run_name: str, device: torch.device):
    config_candidates = _candidate_paths(run_name, "config.json")
    checkpoint_candidates = _candidate_paths(run_name, "best_model.pth")

    config_path = next((path for path in config_candidates if os.path.isfile(path)), None)
    checkpoint_path = next((path for path in checkpoint_candidates if os.path.isfile(path)), None)

    if config_path is None:
        print(
            f"Config not found for {run_name}. Checked: "
            + ", ".join(config_candidates)
        )
        return
    if checkpoint_path is None:
        print(
            f"Checkpoint not found for {run_name}. Checked: "
            + ", ".join(checkpoint_candidates)
        )
        return

    with open(config_path, "r", encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)

    print("\n" + "=" * 80)
    print(f"Evaluating no-augmentation run: {run_name}")
    print("=" * 80)

    model = GCLModel(
        encoder_name=config["encoder_name"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        projection_dim=config["projection_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    run_output_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    dataset_results = {}
    ogb_datasets = base_eval.load_ogb_datasets()

    for dataset_name, ogb_dataset, split_idx, dataset_dir in ogb_datasets:
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        try:
            smiles_map = base_eval.load_smiles_mapping(dataset_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue

        rdkit_splits, skip_stats = base_eval.build_rdkit_feature_aligned_splits(
            ogb_dataset, split_idx, smiles_map
        )

        for split_name in ["train", "valid", "test"]:
            split_dataset = rdkit_splits.get(split_name)
            kept = len(split_dataset) if split_dataset is not None else 0
            skipped = skip_stats.get(split_name, 0)
            print(f"{split_name.capitalize()} split -> kept {kept} graphs, skipped {skipped} graphs")

        missing = [name for name in ["train", "valid", "test"] if len(rdkit_splits.get(name, [])) == 0]
        if missing:
            print(f"{dataset_name} has no RDKit graphs for {', '.join(missing)}; skipping dataset.")
            continue

        train_loader = DataLoader(rdkit_splits["train"], batch_size=256, shuffle=False)
        valid_loader = DataLoader(rdkit_splits["valid"], batch_size=256, shuffle=False)
        test_loader = DataLoader(rdkit_splits["test"], batch_size=256, shuffle=False)

        print("\nExtracting embeddings from train set...")
        train_embeddings, train_labels = base_eval.extract_embeddings(model, train_loader, device)
        print("Extracting embeddings from valid set...")
        valid_embeddings, valid_labels = base_eval.extract_embeddings(model, valid_loader, device)
        print("Extracting embeddings from test set...")
        test_embeddings, test_labels = base_eval.extract_embeddings(model, test_loader, device)

        print("\n" + "=" * 80)
        print("Evaluating classification performance...")
        print("=" * 80)

        print("\nValidation set results:")
        valid_results = base_eval.evaluate_classification(
            train_embeddings, train_labels, valid_embeddings, valid_labels
        )
        for key, value in valid_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        print("\nTest set results:")
        test_results = base_eval.evaluate_classification(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
        for key, value in test_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        dataset_results[dataset_name] = {"valid": valid_results, "test": test_results}

        embeddings_path = os.path.join(run_output_dir, f"{dataset_name}_embeddings.npz")
        base_eval.np.savez(
            embeddings_path,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )
        print(f"Embeddings saved to {embeddings_path}")

    results_path = os.path.join(run_output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(dataset_results, results_file, indent=4)
    print(f"\nAggregated results saved to {results_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for run_name in RUNS:
        evaluate_run(run_name, device)

if __name__ == "__main__":
    main()

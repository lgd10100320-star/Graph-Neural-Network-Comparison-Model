import csv
import gzip
import json
import os

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.gcl_model import GCLModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "results", "no_aug_checkpoints", "gcn_no_aug_config.json")

ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
EDGE_ATTR_DIM = len(BOND_TYPES) + 2


class GraphListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(ROOT_DIR, path))


def get_atom_features(atom):
    features = [1 if atom.GetSymbol() == atom_type else 0 for atom_type in ATOM_TYPES]
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.extend([1 if atom.GetHybridization() == hybridization else 0 for hybridization in HYBRIDIZATION_TYPES])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.GetTotalNumHs())
    return features


def get_bond_features(bond):
    features = [1 if bond.GetBondType() == bond_type else 0 for bond_type in BOND_TYPES]
    features.append(1 if bond.IsInRing() else 0)
    features.append(1 if bond.GetIsConjugated() else 0)
    return features


def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_features = get_bond_features(bond)
            edge_indices.extend([[begin, end], [end, begin]])
            edge_attrs.extend([bond_features, bond_features])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float)

        global_features = torch.tensor(
            [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)],
            dtype=torch.float,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            smiles=smiles,
        )
    except Exception as exc:
        print(f"RDKit failed to process SMILES {smiles}: {exc}")
        return None


def load_smiles_mapping(dataset_dir):
    mapping_path = os.path.join(dataset_dir, "mapping", "mol.csv.gz")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    smiles_map = {}
    with gzip.open(mapping_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError(f"File {mapping_path} is missing the 'smiles' column")
        for row_idx, row in enumerate(reader):
            smiles = row.get("smiles", "").strip()
            smiles_map[row_idx] = smiles if smiles else None

    return smiles_map


def build_rdkit_feature_aligned_splits(ogb_dataset, split_idx, smiles_map):
    cache = {}
    rdkit_splits = {}
    skip_stats = {}

    for split_name, indices in split_idx.items():
        index_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        processed = []
        skipped = 0

        for idx in index_list:
            idx = int(idx)
            smiles = smiles_map.get(idx)
            if smiles is None:
                skipped += 1
                continue

            if idx not in cache:
                graph_data = smiles_to_graph(smiles)
                if graph_data is None:
                    cache[idx] = None
                else:
                    ogb_data = ogb_dataset[idx]
                    graph_data.y = ogb_data.y.clone() if hasattr(ogb_data.y, "clone") else ogb_data.y
                    graph_data.idx = idx
                    cache[idx] = graph_data

            graph_data = cache[idx]
            if graph_data is None:
                skipped += 1
                continue

            processed.append(graph_data)

        rdkit_splits[split_name] = GraphListDataset(processed)
        skip_stats[split_name] = skipped

    return rdkit_splits, skip_stats


def load_ogb_datasets(dataset_names=None):
    import torch.serialization
    from torch_geometric.data.data import DataEdgeAttr

    try:
        torch.serialization.add_safe_globals([DataEdgeAttr])
    except Exception:
        pass

    if dataset_names is None:
        dataset_names = [
            "ogbg-molhiv",
            "ogbg-molpcba",
            "ogbg-moltox21",
            "ogbg-moltoxcast",
            "ogbg-molbace",
            "ogbg-molbbbp",
            "ogbg-molclintox",
            "ogbg-molmuv",
            "ogbg-molsider",
        ]

    root_dir = os.path.join(ROOT_DIR, "data", "ogb")
    os.makedirs(root_dir, exist_ok=True)

    datasets = []
    for dataset_name in dataset_names:
        print(f"Loading OGB dataset: {dataset_name}")
        dataset_dir = os.path.join(root_dir, dataset_name.replace("-", "_"))
        dataset = PygGraphPropPredDataset(name=dataset_name, root=root_dir)
        split_idx = dataset.get_idx_split()
        datasets.append((dataset_name, dataset, split_idx, dataset_dir))

    return datasets


def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting embeddings"):
            batch_data = batch_data.to(device)
            embedding = model.get_embedding(batch_data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(batch_data.y.cpu().numpy())

    return np.vstack(embeddings), np.vstack(labels)


def evaluate_classification(train_embeddings, train_labels, test_embeddings, test_labels):
    print("Training logistic regression classifier...")

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    if train_labels.ndim == 1:
        train_labels = train_labels.reshape(-1, 1)
    if test_labels.ndim == 1:
        test_labels = test_labels.reshape(-1, 1)

    if train_labels.shape[1] > 1:
        results = {}
        for task_idx in range(train_labels.shape[1]):
            train_mask = ~np.isnan(train_labels[:, task_idx])
            test_mask = ~np.isnan(test_labels[:, task_idx])
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            train_targets = train_labels[train_mask, task_idx]
            test_targets = test_labels[test_mask, task_idx]
            if len(np.unique(train_targets)) < 2:
                continue

            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(train_embeddings[train_mask], train_targets)

            predictions = classifier.predict(test_embeddings[test_mask])
            probabilities = classifier.predict_proba(test_embeddings[test_mask])[:, 1]
            accuracy = accuracy_score(test_targets, predictions)
            auc_roc = roc_auc_score(test_targets, probabilities) if len(np.unique(test_targets)) > 1 else 0.0
            results[f"task_{task_idx}"] = {"accuracy": accuracy, "auc_roc": auc_roc}

        if not results:
            return {"avg_accuracy": 0.0, "avg_auc_roc": 0.0, "per_task": {}}

        avg_accuracy = float(np.mean([value["accuracy"] for value in results.values()]))
        avg_auc_roc = float(np.mean([value["auc_roc"] for value in results.values()]))
        return {"avg_accuracy": avg_accuracy, "avg_auc_roc": avg_auc_roc, "per_task": results}

    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()
    train_mask = ~np.isnan(train_labels)
    test_mask = ~np.isnan(test_labels)
    train_targets = train_labels[train_mask]
    test_targets = test_labels[test_mask]

    if len(train_targets) == 0 or len(test_targets) == 0 or len(np.unique(train_targets)) < 2:
        return {"accuracy": 0.0, "auc_roc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings[train_mask], train_targets)

    predictions = classifier.predict(test_embeddings[test_mask])
    probabilities = classifier.predict_proba(test_embeddings[test_mask])[:, 1]
    accuracy = accuracy_score(test_targets, predictions)
    f1 = f1_score(test_targets, predictions, average="binary")
    precision = precision_score(test_targets, predictions, average="binary", zero_division=0)
    recall = recall_score(test_targets, predictions, average="binary", zero_division=0)
    auc_roc = roc_auc_score(test_targets, probabilities) if len(np.unique(test_targets)) > 1 else 0.0

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    run_name = config.get("run_name", config["encoder_name"])
    save_dir = resolve_project_path(config["save_dir"])

    print("Loading trained model...")
    model = GCLModel(
        encoder_name=config["encoder_name"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        projection_dim=config["projection_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    model_path = os.path.join(save_dir, f"{run_name}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    print("\n" + "=" * 80)
    print("Evaluating on OGB datasets...")
    print("=" * 80)

    dataset_results = {}
    ogb_datasets = load_ogb_datasets()

    for dataset_name, ogb_dataset, split_idx, dataset_dir in ogb_datasets:
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print("-" * 80)

        try:
            smiles_map = load_smiles_mapping(dataset_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue

        rdkit_splits, skip_stats = build_rdkit_feature_aligned_splits(ogb_dataset, split_idx, smiles_map)

        for split_name in ["train", "valid", "test"]:
            split_dataset = rdkit_splits.get(split_name)
            kept = len(split_dataset) if split_dataset is not None else 0
            skipped = skip_stats.get(split_name, 0)
            print(f"{split_name.capitalize()} split -> kept {kept} graphs, skipped {skipped} graphs")

        missing_splits = [name for name in ["train", "valid", "test"] if len(rdkit_splits.get(name, [])) == 0]
        if missing_splits:
            print(f"{dataset_name} has no usable RDKit graphs for {', '.join(missing_splits)}; skipping dataset.")
            continue

        train_loader = DataLoader(rdkit_splits["train"], batch_size=256, shuffle=False)
        valid_loader = DataLoader(rdkit_splits["valid"], batch_size=256, shuffle=False)
        test_loader = DataLoader(rdkit_splits["test"], batch_size=256, shuffle=False)

        print("\nExtracting embeddings from train set...")
        train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
        print("Extracting embeddings from valid set...")
        valid_embeddings, valid_labels = extract_embeddings(model, valid_loader, device)
        print("Extracting embeddings from test set...")
        test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

        print("\n" + "=" * 80)
        print("Evaluating classification performance...")
        print("=" * 80)

        print("\nValidation set results:")
        valid_results = evaluate_classification(train_embeddings, train_labels, valid_embeddings, valid_labels)
        for key, value in valid_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        print("\nTest set results:")
        test_results = evaluate_classification(train_embeddings, train_labels, test_embeddings, test_labels)
        for key, value in test_results.items():
            if isinstance(value, dict):
                continue
            print(f"{key}: {value:.4f}")

        dataset_results[dataset_name] = {"valid": valid_results, "test": test_results}

        embeddings_path = os.path.join(save_dir, f"{run_name}_{dataset_name}_embeddings.npz")
        np.savez(
            embeddings_path,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            valid_embeddings=valid_embeddings,
            valid_labels=valid_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )
        print(f"Embeddings saved to {embeddings_path}")

    results_path = os.path.join(save_dir, f"{run_name}_evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_results, handle, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

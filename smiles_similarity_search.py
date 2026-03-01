"""基于训练好的 GIN 模型，在两个 SMILES 文本之间执行相似性检索。"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch, Data

from models.gcl_model import GCLModel

# === 路径与阈值配置（可直接修改） ===
LIBRARY_SMILES_PATH = r"C:\Users\LGD\Desktop\cursor\GNN-12.10\molecular_gnn_final\data\1_million.txt"
QUERY_SMILES_PATH = r"C:\Users\LGD\Desktop\cursor\GNN-12.10\molecular_gnn_final\data\25.5_PCE_smiles.txt"
MODEL_PATH = r"results\checkpoints\gcn_augmentations_0_2_best_model.pth"
OUTPUT_TXT_PATH = r"results/similarity/Filtered_results_0.95_1_million.txt"
DEFAULT_SIM_THRESHOLD = 0.95

# === 模型结构配置（与训练保持一致） ===
MODEL_CONFIG = {
    "encoder_name": "gcn",
    "input_dim": 19,
    "hidden_dim": 256,
    "projection_dim": 128,
    "num_layers": 5,
    "dropout": 0.5,
    "temperature": 0.1,
}

# === 特征配置 ===
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


@dataclass
class MoleculeRecord:
    """用于在读取/转换阶段携带 SMILES 与 RDKit 分子对象。"""

    mol_id: str
    smiles: str
    mol: Chem.Mol


def auto_prefix(path: str, fallback: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base if base else fallback


def read_smiles_file(path: str, prefix: str, progress_label: Optional[str] = None, progress_interval: int = 1000) -> List[MoleculeRecord]:
    records: List[MoleculeRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            lowered = raw.lower()
            if lowered in {"smiles", "pid", "pid smiles", "pid,smiles", "pid\tsmiles"}:
                continue
            pid: Optional[str] = None
            smiles = raw
            if "\t" in raw:
                pid_part, smiles_part = raw.split("\t", 1)
                pid = pid_part.strip() or None
                smiles = smiles_part.strip()
            elif "," in raw:
                pid_part, smiles_part = raw.split(",", 1)
                pid = pid_part.strip() or None
                smiles = smiles_part.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"警告: 无法解析第 {idx + 1} 行 SMILES，已跳过")
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                print(f"警告: 第 {idx + 1} 行分子标准化失败，已跳过")
                continue
            mol_id = pid if pid else f"{prefix}_{idx:05d}"
            records.append(MoleculeRecord(mol_id=mol_id, smiles=Chem.MolToSmiles(mol), mol=mol))
            if progress_label and (idx + 1) % progress_interval == 0:
                print(f"{progress_label}: 已处理 {idx + 1} 行")
    if progress_label:
        print(f"{progress_label}: 读取完成，共 {len(records)} 条有效分子")
    return records


def atom_feature_vector(atom: Chem.Atom) -> List[float]:
    features: List[float] = []
    symbol = atom.GetSymbol()
    features.extend(1.0 if symbol == t else 0.0 for t in ATOM_TYPES)
    features.append(float(atom.GetDegree()))
    features.append(float(atom.GetFormalCharge()))
    features.extend(1.0 if atom.GetHybridization() == h else 0.0 for h in HYBRIDIZATION_TYPES)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetTotalNumHs()))
    return features


def bond_feature_vector(bond: Chem.Bond) -> List[float]:
    features: List[float] = []
    features.extend(1.0 if bond.GetBondType() == t else 0.0 for t in BOND_TYPES)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    return features


def mol_to_graph(record: MoleculeRecord) -> Optional[Data]:
    mol = record.mol
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_features = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond)
        edge_indices.extend([(begin, end), (end, begin)])
        edge_attrs.extend([feat, feat])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mol_id = record.mol_id
    data.smiles = record.smiles
    return data


def molecules_to_graphs(
    records: Sequence[MoleculeRecord],
    progress_label: Optional[str] = None,
    progress_interval: int = 10000,
) -> List[Data]:
    graphs: List[Data] = []
    for record in records:
        graph = mol_to_graph(record)
        if graph is not None:
            graphs.append(graph)
        if progress_label and len(graphs) % progress_interval == 0:
            print(f"{progress_label}: 已构建 {len(graphs)} 个图")
    if progress_label:
        print(f"{progress_label}: 完成，共 {len(graphs)} 个图")
    return graphs


def build_model(model_path: str, device: torch.device) -> GCLModel:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

    model = GCLModel(
        encoder_name=MODEL_CONFIG["encoder_name"],
        input_dim=MODEL_CONFIG["input_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        projection_dim=MODEL_CONFIG["projection_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        dropout=MODEL_CONFIG["dropout"],
        temperature=MODEL_CONFIG["temperature"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_embeddings(
    model: GCLModel,
    graphs: Sequence[Data],
    device: torch.device,
    batch_size: int,
    progress_label: Optional[str] = None,
) -> torch.Tensor:
    if not graphs:
        hidden_dim = MODEL_CONFIG["hidden_dim"]
        return torch.empty((0, hidden_dim))

    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        total = len(graphs)
        for start in range(0, total, batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            embedding = model.get_embedding(batch).detach().cpu()
            embeddings.append(embedding)
            if progress_label:
                processed = min(start + batch_size, total)
                print(f"{progress_label}: 已完成 {processed}/{total}")
    return torch.cat(embeddings, dim=0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_similarity_matrix(library_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
    if library_embeddings.numel() == 0 or query_embeddings.numel() == 0:
        return torch.zeros((library_embeddings.size(0), query_embeddings.size(0)))
    library_norm = F.normalize(library_embeddings, p=2, dim=1)
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    return library_norm @ query_norm.t()


def select_matches(
    sims: torch.Tensor,
    library_records: Sequence[MoleculeRecord],
    top_k: int,
    threshold: float,
    exclude_self: bool,
    query_record: MoleculeRecord,
) -> List[Tuple[MoleculeRecord, float]]:
    matches: List[Tuple[MoleculeRecord, float]] = []
    if sims.numel() == 0:
        return matches

    if top_k > 0:
        k = min(top_k, sims.numel())
        top_vals, top_indices = torch.topk(sims, k=k)
        ordered = zip(top_indices.tolist(), top_vals.tolist())
    else:
        indices = torch.nonzero(sims >= threshold, as_tuple=False).view(-1).tolist()
        ordered = ((idx, sims[idx].item()) for idx in indices)

    for idx, sim in ordered:
        record = library_records[idx]
        if sim <= 0:
            continue
        if threshold > 0 and sim < threshold:
            continue
        if exclude_self and record.mol_id == query_record.mol_id:
            continue
        matches.append((record, float(sim)))

    if top_k <= 0:
        matches.sort(key=lambda item: item[1], reverse=True)

    return matches


def collect_threshold_matches(
    sims: torch.Tensor,
    library_records: Sequence[MoleculeRecord],
    threshold: float,
    exclude_self: bool,
    query_record: MoleculeRecord,
) -> List[Tuple[MoleculeRecord, float]]:
    matches: List[Tuple[MoleculeRecord, float]] = []
    if sims.numel() == 0:
        return matches

    indices = torch.nonzero(sims >= threshold, as_tuple=False).view(-1).tolist()
    for idx in indices:
        sim = sims[idx].item()
        if sim < threshold:
            continue
        record = library_records[idx]
        if exclude_self and record.mol_id == query_record.mol_id:
            continue
        matches.append((record, sim))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def write_similarity_report(
    output_txt: str,
    query_records: Sequence[MoleculeRecord],
    library_records: Sequence[MoleculeRecord],
    sim_matrix: torch.Tensor,
    threshold: float,
    exclude_self: bool,
) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)
    written = 0
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write(f"筛选阈值: {threshold:.2f}\n")
        handle.write("=" * 60 + "\n\n")
        for q_idx, query in enumerate(query_records):
            sims = sim_matrix[:, q_idx]
            matches = collect_threshold_matches(sims, library_records, threshold, exclude_self, query)
            if not matches:
                continue
            handle.write(f"查询分子: {query.mol_id} | {query.smiles}\n")
            for rank, (record, score) in enumerate(matches, start=1):
                handle.write(
                    f"  {rank:02d}. 库分子: {record.mol_id} | {record.smiles} | 相似度: {score:.4f}\n"
                )
                written += 1
            handle.write("\n")
    return written


def main():
    parser = argparse.ArgumentParser(description="在 train_smiles.txt 中筛选与 B 集相似度达标的分子")
    parser.add_argument("--query_smiles", default=QUERY_SMILES_PATH, help="查询分子 SMILES 文本 (B 集)")
    parser.add_argument("--library_smiles", default=LIBRARY_SMILES_PATH, help="库分子 SMILES 文本 (train_smiles)")
    parser.add_argument("--model_path", default=MODEL_PATH, help="训练得到的模型权重路径")
    parser.add_argument("--output_txt", default=OUTPUT_TXT_PATH, help="输出 TXT 文件路径")
    parser.add_argument("--device", default="auto", help="推理设备，auto 会优先使用 GPU")
    parser.add_argument("--batch_size", type=int, default=256, help="批处理大小，越大吞吐越高但更占显存")
    parser.add_argument("--similarity_threshold", type=float, default=DEFAULT_SIM_THRESHOLD, help="最低保留相似度")
    parser.add_argument("--exclude_self", action="store_true", help="当库和查询相同时是否去除自身匹配")
    args = parser.parse_args()

    if not os.path.exists(args.query_smiles):
        raise FileNotFoundError(f"查询文件不存在: {args.query_smiles}")

    library_path = args.library_smiles
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"库文件不存在: {library_path}")

    device = resolve_device(args.device)
    print(f"使用设备: {device}")

    print("加载模型权重...")
    model = build_model(args.model_path, device)

    query_prefix = auto_prefix(args.query_smiles, "query") + "_q"
    lib_prefix = auto_prefix(library_path, "library") + "_lib"

    print("读取查询分子（SMILES 文本）...")
    query_records = read_smiles_file(
        args.query_smiles,
        query_prefix,
        progress_label="查询分子读取进度",
        progress_interval=500,
    )
    print(f"查询分子数量: {len(query_records)}")

    print("读取库分子（SMILES 文本）...")
    library_records = read_smiles_file(
        library_path,
        lib_prefix,
        progress_label="库分子读取进度",
        progress_interval=2000,
    )
    print(f"库分子数量: {len(library_records)}")

    if not query_records:
        raise RuntimeError("未能从查询文件解析出任何有效分子。")
    if not library_records:
        raise RuntimeError("未能从库文件解析出任何有效分子。")

    reuse_library = os.path.abspath(args.query_smiles) == os.path.abspath(library_path)

    print("构建图数据...")
    query_graphs = molecules_to_graphs(
        query_records,
        progress_label="查询图构建进度",
        progress_interval=100,
    )
    print(f"查询图数量: {len(query_graphs)}")
    if reuse_library:
        library_graphs = query_graphs
        print("库图复用查询图")
    else:
        library_graphs = molecules_to_graphs(
            library_records,
            progress_label="库图构建进度",
            progress_interval=1000,
        )
        print(f"库图数量: {len(library_graphs)}")

    if not query_graphs:
        raise RuntimeError("查询图构建失败。")
    if not library_graphs:
        raise RuntimeError("库图构建失败。")

    print("计算查询分子嵌入...")
    query_embeddings = compute_embeddings(
        model,
        query_graphs,
        device,
        args.batch_size,
        progress_label="查询嵌入计算进度",
    )

    print("计算库分子嵌入...")
    if reuse_library:
        library_embeddings = query_embeddings
        print("库嵌入复用查询嵌入")
    else:
        library_embeddings = compute_embeddings(
            model,
            library_graphs,
            device,
            args.batch_size,
            progress_label="库嵌入计算进度",
        )
        print("库嵌入计算完成")

    print("计算相似度矩阵...")
    sim_matrix = compute_similarity_matrix(library_embeddings, query_embeddings)
    print("相似度矩阵计算完成")

    if args.similarity_threshold <= 0:
        print("警告: 相似度阈值未设置，默认 0.95")
        args.similarity_threshold = DEFAULT_SIM_THRESHOLD

    print(f"写出结果到 {args.output_txt} ...")
    rows = write_similarity_report(
        output_txt=args.output_txt,
        query_records=query_records,
        library_records=library_records,
        sim_matrix=sim_matrix,
        threshold=args.similarity_threshold,
        exclude_self=args.exclude_self,
    )
    print("结果写出完成")
    print(f"满足筛选条件的匹配条目: {rows}")
    print("相似性检索完成。")


if __name__ == "__main__":
    main()

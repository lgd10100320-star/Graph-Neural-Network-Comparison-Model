"""
数据预处理模块
处理SMILES格式的分子数据，转换为图结构
"""

import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
from tqdm import tqdm
import pickle
import os


class MoleculeDataset:
    """分子数据集类"""
    
    def __init__(self, smiles_file, save_dir, max_samples=None):
        """
        Args:
            smiles_file: SMILES文件路径
            save_dir: 处理后数据保存目录
            max_samples: 最大样本数量
        """
        self.smiles_file = smiles_file
        self.save_dir = save_dir
        self.max_samples = max_samples
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 读取SMILES
        self.smiles_list = self._load_smiles()
        
        # 处理数据
        self.data_list = self._process_molecules()
    
    def _load_smiles(self):
        """加载SMILES字符串"""
        with open(self.smiles_file, 'r') as f:
            smiles = [line.strip() for line in f if line.strip()]
        
        if self.max_samples:
            smiles = smiles[:self.max_samples]
        
        return smiles
    
    def _get_atom_features(self, atom):
        """提取原子特征"""
        features = []
        
        # 原子类型 (one-hot编码，考虑常见元素)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
        atom_type = atom.GetSymbol()
        features.extend([1 if atom_type == t else 0 for t in atom_types])
        
        # 度数
        features.append(atom.GetDegree())
        
        # 形式电荷
        features.append(atom.GetFormalCharge())
        
        # 杂化类型
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        features.extend([1 if atom.GetHybridization() == t else 0 for t in hybridization_types])
        
        # 是否在芳香环中
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # 氢原子数
        features.append(atom.GetTotalNumHs())
        
        return features
    
    def _get_bond_features(self, bond):
        """提取键特征"""
        features = []
        
        # 键类型
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        features.extend([1 if bond.GetBondType() == t else 0 for t in bond_types])
        
        # 是否在环中
        features.append(1 if bond.IsInRing() else 0)
        
        # 是否共轭
        features.append(1 if bond.GetIsConjugated() else 0)
        
        return features
    
    def _smiles_to_graph(self, smiles):
        """将SMILES转换为图数据"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 提取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self._get_atom_features(atom))
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # 提取边和边特征
            edge_indices = []
            edge_attrs = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # 无向图，添加双向边
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = self._get_bond_features(bond)
                edge_attrs.extend([bond_feat, bond_feat])
            
            if len(edge_indices) == 0:
                # 单原子分子
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 6), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
            # 计算分子描述符作为全局特征
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            global_features = torch.tensor([mol_weight, logp, tpsa], dtype=torch.float)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=global_features,
                smiles=smiles
            )
            
            return data
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _process_molecules(self):
        """处理所有分子"""
        print(f"Processing {len(self.smiles_list)} molecules...")
        
        data_list = []
        failed_count = 0
        
        for smiles in tqdm(self.smiles_list, desc="Converting SMILES to graphs"):
            data = self._smiles_to_graph(smiles)
            if data is not None:
                data_list.append(data)
            else:
                failed_count += 1
        
        print(f"Successfully processed: {len(data_list)}")
        print(f"Failed: {failed_count}")
        
        # 保存处理后的数据
        save_path = os.path.join(self.save_dir, 'processed_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved processed data to {save_path}")
        
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


def load_ogb_dataset(dataset_names=None):
    """批量加载多个OGB数据集用于测试"""
    from ogb.graphproppred import PygGraphPropPredDataset
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # 项目根目录下的 data 路径，避免误保存到 utils/data
    project_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    if dataset_names is None:
        dataset_names = [
            'ogbg-molhiv',
            'ogbg-molpcba',
            'ogbg-moltox21',
            'ogbg-moltoxcast',
            'ogbg-molbace',
            'ogbg-molbbbp',
            'ogbg-molclintox',
            'ogbg-molmuv',
            'ogbg-molsider',
        ]

    # 修复/兼容不同 PyTorch 版本中与 weights_only 相关的问题
    # 在部分 PyTorch 版本中 `torch.serialization.add_safe_globals` 不存在或不可用，
    # 因此这里先做存在性检查并用 try/except 包裹以保证兼容性。
    import torch.serialization
    from torch_geometric.data.data import DataEdgeAttr
    try:
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([DataEdgeAttr])
    except Exception:
        # 如果不存在或调用失败则忽略该步骤，继续加载 OGB 数据集
        pass

    datasets = []
    for dataset_name in dataset_names:
        print(f"Loading OGB dataset: {dataset_name}")
        dataset_root = os.path.join(project_data_root, 'ogb')
        os.makedirs(dataset_root, exist_ok=True)
        dataset = PygGraphPropPredDataset(name=dataset_name, root=dataset_root)
        split_idx = dataset.get_idx_split()
        datasets.append((dataset_name, dataset, split_idx))

    return datasets


def analyze_dataset(data_list):
    """分析数据集统计信息"""
    num_nodes = [data.x.size(0) for data in data_list]
    num_edges = [data.edge_index.size(1) for data in data_list]
    
    stats = {
        'num_graphs': len(data_list),
        'avg_nodes': np.mean(num_nodes),
        'std_nodes': np.std(num_nodes),
        'min_nodes': np.min(num_nodes),
        'max_nodes': np.max(num_nodes),
        'avg_edges': np.mean(num_edges),
        'std_edges': np.std(num_edges),
        'min_edges': np.min(num_edges),
        'max_edges': np.max(num_edges),
        'node_feature_dim': data_list[0].x.size(1),
        'edge_feature_dim': data_list[0].edge_attr.size(1) if data_list[0].edge_attr.size(0) > 0 else 0,
    }
    
    return stats


if __name__ == "__main__":
    # 处理训练数据
    print("=" * 80)
    print("Processing training data from SMILES file...")
    print("=" * 80)
    
    train_dataset = MoleculeDataset(
        smiles_file=r'C:\Users\LGD\Desktop\cursor代码\molecular_gnn_final\data\train_smiles.txt',
        save_dir=r'C:\Users\LGD\Desktop\cursor代码\molecular_gnn_final\data\processed',
        max_samples=100000
    )
    
    # 分析数据集
    stats = analyze_dataset(train_dataset.data_list)
    print("\nDataset Statistics:")
    print("-" * 80)
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # 保存统计信息
    with open(r'C:\Users\LGD\Desktop\cursor代码\molecular_gnn_final\data\processed\dataset_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print("\n" + "=" * 80)
    print("Loading OGB test datasets...")
    print("=" * 80)
    
    # 加载多个OGB数据集
    ogb_datasets = load_ogb_dataset()
    
    for dataset_name, ogb_dataset, split_idx in ogb_datasets:
        print(f"\nOGB Dataset Info ({dataset_name}):")
        print(f"Total graphs: {len(ogb_dataset)}")
        print(f"Train: {len(split_idx['train'])}")
        print(f"Valid: {len(split_idx['valid'])}")
        print(f"Test: {len(split_idx['test'])}")
        print(f"Task type: {ogb_dataset.task_type}")
        print(f"Number of tasks: {ogb_dataset.num_tasks}")
    
    print("\nData preprocessing completed!")


import torch
import random

def node_masking(data, mask_rate=0.15):
    """随机掩码一部分节点特征"""
    new_data = data.clone()
    num_nodes = new_data.x.size(0)
    mask = torch.rand(num_nodes) < mask_rate
    new_data.x[mask] = 0  # 或者替换为特殊的掩码标记
    return new_data

def edge_deletion(data, delete_rate=0.15):
    """随机删除一部分边"""
    new_data = data.clone()
    num_edges = new_data.edge_index.size(1)
    keep_mask = torch.rand(num_edges // 2) > delete_rate # 因为边是双向的
    
    # 保持双向
    keep_mask = keep_mask.repeat(2)
    
    new_data.edge_index = new_data.edge_index[:, keep_mask]
    if new_data.edge_attr is not None:
        new_data.edge_attr = new_data.edge_attr[keep_mask]
        
    return new_data

def subgraph_deletion(data, delete_rate=0.15):
    """随机删除一个子图（通过随机游走）"""
    new_data = data.clone()
    num_nodes = new_data.num_nodes
    
    if num_nodes <= 1:
        return new_data

    # 随机选择一个起始节点
    start_node = random.randint(0, num_nodes - 1)
    
    # 计算要删除的节点数
    num_delete_nodes = int(num_nodes * delete_rate)
    if num_delete_nodes == 0:
        return new_data

    # 从起始节点开始进行随机游走，收集要删除的节点
    to_delete = {start_node}
    current_node = start_node
    
    for _ in range(num_delete_nodes * 2): # 游走步数，保证能找到足够节点
        neighbors = new_data.edge_index[1, new_data.edge_index[0] == current_node]
        if len(neighbors) == 0:
            # 如果没有邻居，随机跳到另一个节点
            current_node = random.randint(0, num_nodes - 1)
            continue
        
        current_node = neighbors[random.randint(0, len(neighbors) - 1)].item()
        to_delete.add(current_node)
        
        if len(to_delete) >= num_delete_nodes:
            break

    # 创建节点掩码
    node_mask = torch.ones(num_nodes, dtype=torch.bool)
    node_mask[list(to_delete)] = False
    
    # 根据掩码过滤节点和边
    new_data = new_data.subgraph(node_mask)
    
    return new_data


class Augmentation:
    """数据增强组合器"""
    def __init__(self, methods, rates):
        self.methods = methods
        self.rates = rates
        self.aug_fn_map = {
            'node_masking': node_masking,
            'edge_deletion': edge_deletion,
            'subgraph_deletion': subgraph_deletion
        }

    def __call__(self, data):
        """对给定的图数据应用两种随机增强"""
        # 随机选择两种不同的增强方法
        aug1_name, aug2_name = random.sample(self.methods, 2)
        
        aug1_rate = self.rates.get(aug1_name, 0.15)
        aug2_rate = self.rates.get(aug2_name, 0.15)
        
        data1 = self.aug_fn_map[aug1_name](data, aug1_rate)
        data2 = self.aug_fn_map[aug2_name](data, aug2_rate)
        
        return data1, data2

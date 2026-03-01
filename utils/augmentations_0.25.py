"""数据增强预设：所有增强比例 = 0.25

训练脚本会动态加载本文件，读取 AUGMENTATION_RATES。
"""

AUGMENTATION_RATES = {
    "node_masking": 0.25,
    "edge_deletion": 0.25,
    "subgraph_deletion": 0.25,
}


def get_rates():
    return dict(AUGMENTATION_RATES)

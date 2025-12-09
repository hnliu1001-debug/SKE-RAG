import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data_loader import NTUSkeletonDataset
from src.model import PhysicsGuidedAtomicProjector
from tqdm import tqdm
import os

# --- 配置 ---
DATA_DIR = "./data/raw_skeletons"
MODEL_WEIGHTS = "projector_weights.pth"
SAVE_PATH = "knowledge_base.pt"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的 11 个类别 (含 Kick)
TARGET_CLASSES = [1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 24]

def load_model():
    print(f"🚀 Loading Projector from {MODEL_WEIGHTS}...")
    model = PhysicsGuidedAtomicProjector(
        in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    return model

def build_index():
    # 1. 加载数据 (只用训练集模式，不增强)
    dataset = NTUSkeletonDataset(DATA_DIR, selected_classes=TARGET_CLASSES, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(dataset) == 0:
        print("❌ Data not found.")
        return

    model = load_model()
    
    kb_vectors = []
    kb_atoms = []
    kb_labels = []
    
    print(f"📚 Indexing {len(dataset)} samples into Knowledge Base...")
    
    with torch.no_grad():
        for data, target in tqdm(loader):
            data = data.to(DEVICE)
            
            # Forward: 获取高维向量 和 原子评分
            # atomic_vectors: [B, 7, 192]
            atomic_vectors, _, _, _ = model(data)
            
            # 1. 高维向量 (用于检索) -> Flatten
            flat_vecs = atomic_vectors.view(atomic_vectors.size(0), -1).cpu()
            
            # 2. 原子评分 (用于描述) -> Norm
            atom_scores = torch.norm(atomic_vectors, p=2, dim=2).cpu()
            
            kb_vectors.append(flat_vecs)
            kb_atoms.append(atom_scores)
            kb_labels.append(target)
            
    # 合并
    kb_data = {
        "vectors": torch.cat(kb_vectors, dim=0), # [N, 1344]
        "atoms": torch.cat(kb_atoms, dim=0),     # [N, 7]
        "labels": torch.cat(kb_labels, dim=0)    # [N]
    }
    
    torch.save(kb_data, SAVE_PATH)
    print(f"✅ Knowledge Base saved to {SAVE_PATH}")
    print(f"   Shape: {kb_data['vectors'].shape}")

if __name__ == "__main__":
    build_index()
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import PhysicsGuidedAtomicProjector
from src.data_loader import read_ntu_skeleton, preprocess_sample
import os

TEST_FILE_PATH = r"E:\111shiyan\datasets\nturgb+d_skeletons\S017C002P008R001A010.skeleton"
MODEL_WEIGHTS = "projector_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = PhysicsGuidedAtomicProjector(
        in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    
    raw_data = read_ntu_skeleton(TEST_FILE_PATH)
    if raw_data is None: return
    data = preprocess_sample(raw_data, target_frames=50)
    inp = torch.FloatTensor(data).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # 只取第一个返回值
        atomic_vectors, _, _, _ = model(inp)
        
    intensities = torch.norm(atomic_vectors, p=2, dim=2).squeeze().cpu().numpy()
    # 归一化仅用于绘图，不用于RAG逻辑
    if intensities.max() > 0:
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        
    atom_labels = ["Posture", "Legs", "Hand-Face", "Upper-Exp", "Symmetry", "Micro-Mo", "Physics"]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True, facecolor='#050510')
    
    values = np.concatenate((intensities, [intensities[0]]))
    angles = np.linspace(0, 2 * np.pi, len(atom_labels), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, color='#00FFFF', linewidth=3)
    ax.fill(angles, values, color='#00FFFF', alpha=0.35)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(atom_labels, size=14, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
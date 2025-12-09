import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import NTUSkeletonDataset
from src.model import PhysicsGuidedAtomicProjector, AtomicActionClassifier
import os

# --- 配置 ---
DATA_DIR = "./data/raw_skeletons"
BATCH_SIZE = 16 
EPOCHS = 40     
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 包含 A024 Kick，共 11 类
TARGET_CLASSES = [1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 24] 

def main():
    if not os.path.exists(DATA_DIR): 
        print(f"❌ 错误: {DATA_DIR} 不存在")
        return
        
    # 开启数据增强
    dataset = NTUSkeletonDataset(DATA_DIR, selected_classes=TARGET_CLASSES, augment=True)
    if len(dataset) == 0: 
        print("❌ 数据集为空")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型 (V4.0)
    projector = PhysicsGuidedAtomicProjector(
        in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25, lambda_reg=0.2
    ).to(DEVICE)
    
    model = AtomicActionClassifier(projector, num_classes=len(TARGET_CLASSES)).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Starting Training (Ske-RAG V4.0 Compatible)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            
            # 【关键修改】这里使用 _ 忽略掉第 5 个返回值 (vecs)
            # V4.0 model returns: logits, phy_loss, s_attn, t_weights, vecs
            logits, phy_loss, _, _, _ = model(data)
            
            cls_loss = criterion(logits, target)
            loss = cls_loss + phy_loss 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    torch.save(projector.state_dict(), "projector_weights.pth")
    print("💾 Model saved to projector_weights.pth")

if __name__ == "__main__":
    main()
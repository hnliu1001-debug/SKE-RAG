import torch
from torch.utils.data import DataLoader
from src.data_loader import NTUSkeletonDataset
from src.model import PhysicsGuidedAtomicProjector
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import os
import pandas as pd

# --- 配置 ---
DATA_DIR = "./data/raw_skeletons2"  # 请确保路径正确
MODEL_WEIGHTS = "projector_weights.pth"
OUTPUT_DIR = "./output"            # 结果保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 11类定义 (含 Kick)
TARGET_CLASSES = [1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 24]
CLASS_NAMES = [
    "Drink", "Eat", "Brush", 
    "Pickup", "Throw", 
    "Sit", "Stand", 
    "Clap", "Write", "Tear", 
    "Kick"
]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 Created output directory: {directory}")

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, filename="confusion_matrix.png"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fmt = 'd'
    title = "Confusion Matrix"
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized Confusion Matrix"

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True)
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title(title, fontweight='bold', size=15)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_tsne(X, y, classes, filename="tsne_visualization.png"):
    """绘制 t-SNE 聚类图"""
    print("⏳ Running t-SNE (this might take a while)...")
    
    # 降维
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 构建 DataFrame 方便绘图
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'label': [classes[i] for i in y]
    })
    
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df, x='x', y='y', hue='label', 
        palette="tab20", style='label', s=80, alpha=0.8
    )
    
    plt.title("t-SNE Projection of Atomic Features", fontweight='bold', size=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Actions")
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_class_accuracy(y_true, y_pred, classes, filename="class_accuracy.png"):
    """绘制各类别的准确率柱状图"""
    cm = confusion_matrix(y_true, y_pred)
    # 对角线元素 / 该行总和
    acc = cm.diagonal() / cm.sum(axis=1)
    
    # 排序（可选，让图更好看）
    sorted_indices = np.argsort(acc)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_acc = acc[sorted_indices]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_classes, sorted_acc, color=sns.color_palette("viridis", len(classes)))
    
    plt.ylim(0, 1.1)
    plt.axhline(y=np.mean(acc), color='r', linestyle='--', label=f'Mean Acc: {np.mean(acc):.2f}')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel('Accuracy', fontweight='bold')
    plt.title("Per-Class Recognition Accuracy", fontweight='bold', size=14)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 1. 加载数据
    dataset = NTUSkeletonDataset(DATA_DIR, selected_classes=TARGET_CLASSES, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. 加载模型
    model = PhysicsGuidedAtomicProjector(
        in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25
    ).to(DEVICE)
    
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    else:
        print(f"❌ Error: Weights {MODEL_WEIGHTS} not found.")
        return
        
    model.eval()
    
    # 3. 提取特征
    print("🚀 Extracting features...")
    all_X, all_y = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            # Projector 返回 4 个值: vecs, attn, loss, weights
            # 如果报错，请检查 model.py 的 forward 返回值数量
            outputs = model(data)
            vecs = outputs[0] # 取第一个返回值
            
            all_X.append(vecs.view(vecs.size(0), -1).cpu().numpy())
            all_y.append(target.numpy())
            
    if not all_X:
        print("❌ No data loaded.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # 4. 线性分类器评估
    print("🧠 Training Linear Probe...")
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    # 打印文字报告
    acc = accuracy_score(y, y_pred)
    print(f"\n🏆 Overall Accuracy: {acc:.4f}")
    print("\n" + classification_report(y, y_pred, target_names=CLASS_NAMES))
    
    # 5. 生成论文图表
    print("\n🎨 Generating Plots...")
    
    # 图 1: 混淆矩阵 (计数)
    plot_confusion_matrix(y, y_pred, CLASS_NAMES, normalize=False, filename="confusion_matrix_count.png")
    
    # 图 2: 混淆矩阵 (百分比 - 推荐论文用)
    plot_confusion_matrix(y, y_pred, CLASS_NAMES, normalize=True, filename="confusion_matrix_norm.png")
    
    # 图 3: 各类别准确率柱状图
    plot_class_accuracy(y, y_pred, CLASS_NAMES, filename="class_accuracy_bar.png")
    
    # 图 4: t-SNE 特征分布图
    # 如果样本太多 (>5000)，可以考虑先采样，否则太慢
    if len(X) > 10:
        plot_tsne(X, y, CLASS_NAMES, filename="feature_tsne.png")
    else:
        print("⚠️ Not enough samples for t-SNE.")

    print(f"\n✨ All results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.model import PhysicsGuidedAtomicProjector
from src.data_loader import read_ntu_skeleton, preprocess_sample
import os
from collections import Counter

# --- 🛠️ 配置区域 ---
# 替换为你想测试的文件路径
# 建议测试: A010(Clap), A001(Drink), A027(Jump/OOD)
TEST_FILE = r"E:\111shiyan\datasets\nturgb+d_skeletons\S015C003P025R001A012.skeleton"

KB_PATH = "knowledge_base.pt"
MODEL_WEIGHTS = "projector_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的训练集类别 (用于显示邻居名字)
TRAINED_CLASSES = [
    "Drink Water", "Eat Meal", "Brush Teeth", 
    "Pickup", "Throw", 
    "Sit Down", "Stand Up", 
    "Clapping", "Writing", "Tear Paper",
    "Kicking"
]

# 7个原子定义
ATOM_NAMES = [
    "Posture Stability", "Leg Movement", "Hand-Face Interaction", 
    "Upper Body Expansion", "Symmetry", "Micro-Motion (Tremor)", "Physics (Height/Ext)"
]

def load_resources():
    """加载模型和知识库"""
    # num_atoms=6 (隐式) + 1 (显式) = 7 (模型内部处理)
    model = PhysicsGuidedAtomicProjector(
        in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25
    ).to(DEVICE)
    
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    else:
        print(f"❌ Weights not found: {MODEL_WEIGHTS}")
        exit()
    model.eval()
    
    if not os.path.exists(KB_PATH):
        print("❌ Knowledge Base not found. Run build_kb.py first.")
        exit()
    kb = torch.load(KB_PATH)
    return model, kb

def calculate_real_physics(data_tensor):
    """
    【核心】计算真实的物理指标 (Ground Truth Physics)
    这些值是绝对可靠的，用于纠正神经网络的幻觉。
    """
    x = data_tensor # [1, 3, T, 25]
    head = x[:, :, :, 3]
    l_hand = x[:, :, :, 7]
    r_hand = x[:, :, :, 11]
    
    # 1. 垂直高度差 (Hand Y - Head Y)
    # 负数表示手在头下面。例如 -0.3m 通常是胸口位置。
    avg_hand_y = (l_hand[:, 1, :] + r_hand[:, 1, :]) / 2.0
    head_y = head[:, 1, :]
    height_diff = (avg_hand_y - head_y).mean().item()
    
    # 2. 真实手头距离 (Euclidean Distance)
    # 如果 > 0.25m，物理上就不可能是刷牙/吃饭。
    d_lh = torch.norm(l_hand - head, p=2, dim=1)
    d_rh = torch.norm(r_hand - head, p=2, dim=1)
    avg_dist = (d_lh + d_rh).mean().item() / 2.0
    
    return height_diff, avg_dist

def get_atom_description(scores):
    """生成原子特征描述 (不使用MinMax归一化，使用绝对阈值)"""
    desc_list = []
    
    for i, score in enumerate(scores):
        name = ATOM_NAMES[i]
        # 根据经验阈值判断强弱
        if score > 0.8: level = "VERY HIGH"
        elif score > 0.5: level = "HIGH"
        elif score < 0.2: level = "LOW"
        else: level = "MODERATE"
        
        # 只报告显著特征
        if level != "MODERATE":
            desc_list.append(f"- {name}: **{level}** ({score:.2f})")
            
    if not desc_list:
        return "No significant atomic features detected."
    return "\n".join(desc_list)

def generate_universal_prompt(atom_text, neighbor_text, sim_scores, real_h_diff, real_h_dist):
    """
    Ske-RAG V4.0: 包含 OOD 检测与 物理冲突仲裁 的通用 Prompt
    """
    # 1. 判定 OOD
    max_sim = sim_scores[0]
    is_ood = max_sim < 0.65
    
    # 2. 判定物理冲突 (Neural vs Real Physics)
    # 例如：模型说 Hand-Face High，但物理距离 > 0.25m
    conflict_flag = ""
    if "Hand-Face" in atom_text and real_h_dist > 0.25:
        conflict_flag = """
**⚠️ PHYSICAL CONFLICT DETECTED**: 
The Neural Model detected 'Hand-Face Interaction', BUT the Explicit Physical Sensor shows hands are **> 0.25m away from head**.
-> This is likely a False Positive. The action is probably near the Chest (e.g., Clapping), not the Face.
"""

    # --- Prompt 组装 ---
    prompt = f"""
[Role]
You are a Forensic Biomechanics Expert. Your goal is to identify a human action by triangulating "Neural Intuition", "Hard Physics", and "Database Evidence".

[1. Neural Atomic Observations (Subjective)]
The deep learning model detected the following patterns:
{atom_text}

[2. Explicit Physical Measurements (Objective Truth)]
- Hand-Head Vertical Diff: **{real_h_diff:.2f} meters** (Negative = Hands below Head).
- Hand-Head Absolute Distance: **{real_h_dist:.2f} meters**.
{conflict_flag}

[3. Retrieval Evidence (Historical Data)]
Top-5 similar records from the knowledge base:
{neighbor_text}
"""

    if is_ood:
        prompt += f"""
[4. Critical Analysis: UNKNOWN ACTION (OOD)]
**Status**: The maximum similarity ({max_sim:.2f}) is LOW (< 0.65).
**Action**:
1. IGNORE the labels in [Retrieval Evidence] (they are likely incorrect guesses).
2. EXCLUDE all known trained classes: {TRAINED_CLASSES}.
3. SYNTHESIZE the Atomic Observations to infer a new action name.
   - Example: If Legs are VERY HIGH and Physics is HIGH -> Infer "Jumping".
"""
    else:
        prompt += f"""
[4. Final Verdict: KNOWN ACTION]
**Status**: High confidence retrieval ({max_sim:.2f}).
**Action**:
1. If a **Physical Conflict** was detected above (e.g., Hand-Face mismatch), prioritize the **Explicit Physical Measurements** and the **Retrieval Consensus** over the Atomic labels.
2. Example: If Retrieval says "Clapping" and Physics says "Chest Level", conclude "Clapping" despite Atomic errors.
3. Otherwise, trust the consensus of the retrieved neighbors.
"""

    prompt += """
**Output Format**:
- Final Prediction: [Action Name]
- Confidence: [High/Medium/Low]
- Reasoning: [Explain your logic, specifically how you used Physics to verify or reject the Neural Observations]
"""
    return prompt

def run_rag_pipeline():
    model, kb = load_resources()
    
    # 1. 读取数据
    raw_data = read_ntu_skeleton(TEST_FILE)
    if raw_data is None: 
        print(f"❌ Could not read file: {TEST_FILE}")
        return
        
    data = preprocess_sample(raw_data, target_frames=50)
    inp = torch.FloatTensor(data).unsqueeze(0).to(DEVICE)
    
    # 2. 提取特征
    with torch.no_grad():
        # Projector 返回 4 个值
        vecs, _, _, _ = model(inp)
        
    query_vec = vecs.view(1, -1).cpu().numpy()
    query_atoms = torch.norm(vecs, p=2, dim=2).squeeze().cpu().numpy()
    
    # 3. 检索
    kb_vecs = kb['vectors'].numpy()
    sims = cosine_similarity(query_vec, kb_vecs)[0]
    
    # 负步长修复 (.copy())
    top_k_idxs = np.argsort(sims)[-5:][::-1].copy()
    
    top_k_sims = sims[top_k_idxs]
    top_k_labels = kb['labels'][top_k_idxs].numpy()
    
    # 4. 计算真实物理
    real_h_diff, real_h_dist = calculate_real_physics(inp)
    
    # 5. 生成 Prompt
    print(f"\n======== 🧠 Ske-RAG Reasoning Engine ========")
    
    atom_text = get_atom_description(query_atoms)
    
    neighbor_text = ""
    for i, idx in enumerate(top_k_idxs):
        lbl_idx = int(top_k_labels[i])
        # 防止索引越界
        if lbl_idx < len(TRAINED_CLASSES):
            lbl_name = TRAINED_CLASSES[lbl_idx]
        else:
            lbl_name = f"Unknown_Class_{lbl_idx}"
            
        neighbor_text += f"{i+1}. **{lbl_name}** (Sim: {top_k_sims[i]:.2f})\n"
    
    prompt = generate_universal_prompt(
        atom_text, 
        neighbor_text, 
        top_k_sims, 
        real_h_diff, 
        real_h_dist
    )
    
    print("\n📋 Copy this prompt to LLM:\n")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

if __name__ == "__main__":
    run_rag_pipeline()
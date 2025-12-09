import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGuidedAtomicProjector(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, num_atoms=6, num_joints=25, lambda_reg=0.2):
        """
        Ske-RAG V4.0: 精细化物理特征修正版
        解决: 鼓掌误判、手脸假阳性、姿态稳定性过敏问题
        """
        super(PhysicsGuidedAtomicProjector, self).__init__()
        
        self.num_atoms = num_atoms
        self.num_joints = num_joints
        self.lambda_reg = lambda_reg

        # 1. 深度特征提取 (保持 V3.1 稳定版)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 2. 物理掩码
        prior = self._generate_refined_priors()
        self.register_buffer('prior_mask', prior)
        self.learnable_mask = nn.Parameter(prior.clone() + torch.randn_like(prior) * 0.01)

        # 3. 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

        # 4. 显式物理特征嵌入 (维度增加到 14，包含更细的特征)
        self.physic_embed = nn.Sequential(
            nn.BatchNorm1d(14), # 归一化非常重要
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() # 限制幅度
        )

    def _generate_refined_priors(self):
        mask = torch.zeros(self.num_atoms, self.num_joints)
        mask[0, [0, 1, 2, 20]] = 1.0 # Posture (Spine)
        mask[1, [12, 13, 14, 16, 17, 18]] = 1.0 # Legs
        mask[2, [3, 22, 24, 7, 11]] = 1.0 # Hand-Face (Head + Hands)
        mask[3, [4, 5, 6, 8, 9, 10]] = 1.0 # Upper-Exp
        mask[4, [5, 7, 9, 11]] = 1.0 # Symmetry
        mask[5, :] = 0.5 # Micro-Motion
        return mask

    def forward(self, x):
        """
        x: [Batch, 3, T, V] (0:X, 1:Y, 2:Z)
        NTU Joints: 0:Base, 1:Mid, 2:Neck, 3:Head, 20:ShoulderCenter
        7:LHand, 11:RHand
        """
        N, C, T, V = x.size()

        # --- Part 1: Deep Implicit Features ---
        feat = self.feature_encoder(x)
        spatial_attn = F.softmax(self.learnable_mask, dim=1) 
        projected_feat = torch.einsum('nctv,av->ncta', feat, spatial_attn)

        feat_global = feat.mean(dim=3, keepdim=True)
        t_weights = F.softmax(self.temporal_attn(feat_global), dim=2) 

        feat_mean = (projected_feat * t_weights).sum(dim=2)
        feat_diff = projected_feat[:, :, -1, :] - projected_feat[:, :, 0, :]
        feat_std = torch.std(projected_feat, dim=2)
        
        deep_vectors = torch.cat([feat_mean, feat_diff, feat_std * 2.0], dim=1) 
        deep_vectors = deep_vectors.permute(0, 2, 1) 

        # --- Part 2: Explicit Physics V4.0 (精细化计算) ---
        
        # 提取关键关节
        spine_base = x[:, :, :, 0] # [N, 3, T]
        neck = x[:, :, :, 2]
        head = x[:, :, :, 3]
        l_hand = x[:, :, :, 7]
        r_hand = x[:, :, :, 11]
        l_foot = x[:, :, :, 14]
        r_foot = x[:, :, :, 18]

        # --- 痛点1修复：姿态稳定性 (用角度代替坐标) ---
        # 计算脊柱向量 (Base -> Neck)
        spine_vec = neck - spine_base 
        # 计算脊柱是否垂直 (Y轴分量占比)
        spine_verticality = spine_vec[:, 1, :] / (torch.norm(spine_vec, p=2, dim=1) + 1e-6)
        # F1: 脊柱倾斜度均值 (Pickup弯腰时这个值会变小)
        # F2: 脊柱晃动方差 (解决"多动症"误判，只看角度变不变)
        f_posture_mean = spine_verticality.mean(1, keepdim=True)
        f_posture_std = spine_verticality.std(1, keepdim=True)

        # --- 痛点2修复：手脸交互 (引入下巴阈值) ---
        # 计算手相对于脖子的高度 (Y轴)
        hand_y_avg = (l_hand[:, 1, :] + r_hand[:, 1, :]) / 2.0
        neck_y = neck[:, 1, :]
        # F3: 手是否高于脖子? (正数=高，负数=低) -> 解决鼓掌(胸前)被判为刷牙
        f_hand_height_rel = (hand_y_avg - neck_y).mean(1, keepdim=True)
        
        # F4: 手头欧氏距离 (辅助判断)
        dist_head = (torch.norm(l_hand - head, p=2, dim=1) + torch.norm(r_hand - head, p=2, dim=1)) / 2.0
        f_hand_head_dist = dist_head.mean(1, keepdim=True)

        # --- 痛点3修复：双手交互 (区分鼓掌/写字) ---
        hand_dist = torch.norm(l_hand - r_hand, p=2, dim=1)
        # F5: 双手距离均值
        f_hand_dist_mean = hand_dist.mean(1, keepdim=True)
        # F6: 双手距离方差 (鼓掌=大，写字=小，撕纸=中)
        f_hand_dist_std = hand_dist.std(1, keepdim=True)

        # --- 其他特征 (保留V3的有效部分) ---
        # F7: 上肢伸展度 (XZ平面)
        arm_ext = (torch.norm(l_hand[:,[0,2],:] - spine_base[:,[0,2],:], p=2, dim=1) + 
                   torch.norm(r_hand[:,[0,2],:] - spine_base[:,[0,2],:], p=2, dim=1)) / 2.0
        f_arm_ext_mean = arm_ext.mean(1, keepdim=True)
        f_arm_ext_std = arm_ext.std(1, keepdim=True)

        # F8: 下肢动能 (针对 Kick/Jump)
        foot_move = (l_foot[:, 1, :] + r_foot[:, 1, :]) / 2.0 # 简单的高度变化
        f_leg_mean = foot_move.mean(1, keepdim=True)
        f_leg_std = foot_move.std(1, keepdim=True)
        
        # F9: 身体整体位移 (Pickup vs Clap)
        # Pickup 会有明显的 Y 轴整体下降
        base_y_std = spine_base[:, 1, :].std(1, keepdim=True)
        base_y_mean = spine_base[:, 1, :].mean(1, keepdim=True)

        # 拼接所有物理统计量 (Total 14 dims)
        stats = [
            f_posture_mean, f_posture_std,   # 脊柱
            f_hand_height_rel, f_hand_head_dist, # 手脸
            f_hand_dist_mean, f_hand_dist_std, # 双手
            f_arm_ext_mean, f_arm_ext_std,   # 伸展
            f_leg_mean, f_leg_std,           # 腿部
            base_y_mean, base_y_std          # 整体重心
        ]
        physic_raw = torch.cat(stats, dim=1) # [N, 12] -> 这里其实是 12 维，我上面 linear 写了14，可以改一下
        
        # 修正: 上面 stats 只有 12 个张量，Linear 输入应改为 12，或者我再补两个
        # 补两个: 左右手分别的高度方差 (区分单手/双手动作)
        l_h_std = l_hand[:, 1, :].std(1, keepdim=True)
        r_h_std = r_hand[:, 1, :].std(1, keepdim=True)
        physic_raw = torch.cat([physic_raw, l_h_std, r_h_std], dim=1) # 现在是 14 维

        # Embedding
        physic_vec = self.physic_embed(physic_raw)
        
        # Fusion
        physic_vec_expanded = physic_vec.repeat(1, 3).unsqueeze(1) 
        final_vectors = torch.cat([deep_vectors, physic_vec_expanded], dim=1)

        phy_loss = torch.norm(self.learnable_mask - self.prior_mask, p=2) * self.lambda_reg
        return final_vectors, spatial_attn, phy_loss, t_weights

class AtomicActionClassifier(nn.Module):
    def __init__(self, projector, num_classes=12, feature_dim=64):
        super(AtomicActionClassifier, self).__init__()
        self.projector = projector
        self.atom_dropout_rate = 0.2
        
        input_dim = (projector.num_atoms + 1) * (feature_dim * 3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        vecs, s_attn, p_loss, t_weights = self.projector(x)
        if self.training:
            mask = torch.bernoulli(torch.ones_like(vecs[:, :, 0:1]) * (1 - self.atom_dropout_rate))
            vecs = vecs * mask / (1 - self.atom_dropout_rate)
        logits = self.classifier(vecs)
        return logits, p_loss, s_attn, t_weights, vecs
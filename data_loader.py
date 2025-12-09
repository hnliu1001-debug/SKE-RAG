import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

def read_ntu_skeleton(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_frames = int(lines[0])
    current_line = 1
    skeleton_sequence = []
    
    for f in range(num_frames):
        num_bodies = int(lines[current_line])
        current_line += 1
        if num_bodies == 0: continue
        current_line += 1 
        num_joints = int(lines[current_line])
        current_line += 1
        frame_joints = []
        for j in range(num_joints):
            coords = list(map(float, lines[current_line].split()[:3]))
            frame_joints.append(coords)
            current_line += 1
        skeleton_sequence.append(frame_joints)
        for b in range(1, num_bodies):
            current_line += 1
            num_j = int(lines[current_line])
            current_line += 1 + num_j
            
    data = np.array(skeleton_sequence)
    if data.shape[0] == 0: return None
    data = data.transpose(2, 0, 1) # [3, T, V]
    return data

def preprocess_sample(data, target_frames=50, augment=False):
    C, T, V = data.shape
    
    if T != target_frames:
        indices = np.linspace(0, T-1, target_frames).astype(int)
        data = data[:, indices, :]
    
    # 【关键】只减去第一帧 SpineBase，保留跳跃/踢腿的绝对位移
    initial_spine_base = data[:, 0:1, 0:1] 
    data = data - initial_spine_base 
    
    if augment:
        noise = np.random.normal(0, 0.005, data.shape)
        data = data + noise
        
    return data

class NTUSkeletonDataset(Dataset):
    def __init__(self, data_dir, selected_classes=None, augment=False):
        self.files = []
        self.labels = []
        self.augment = augment
        all_files = glob.glob(os.path.join(data_dir, "*.skeleton"))
        
        for f in all_files:
            filename = os.path.basename(f)
            try:
                action_class = int(filename[filename.find('A')+1 : filename.find('A')+4])
            except: continue
            
            if selected_classes and action_class in selected_classes:
                self.files.append(f)
                self.labels.append(selected_classes.index(action_class))
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        raw_data = read_ntu_skeleton(path)
        if raw_data is None: return torch.zeros(3, 50, 25), self.labels[idx]
        processed_data = preprocess_sample(raw_data, target_frames=50, augment=self.augment)
        return torch.FloatTensor(processed_data), self.labels[idx]
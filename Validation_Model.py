import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import os

warnings.filterwarnings('ignore', category=FutureWarning)
script_dir = os.path.dirname(os.path.abspath(__file__))
# -------------------- 设置随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# -------------------- 设备配置 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 超参数 --------------------
BATCH_SIZE = 500
EPOCHS = 200
LEARNING_RATE = 1e-3
LAMBDA_THICKNESS = 0.2  # 厚度损失权重
LAMBDA_FITNESS = 1  # 适应度损失权重
D_CNN = 512  # CNN输出特征维度
D_MODEL = 512  # Transformer模型维度
NHEAD = 8  # 多头注意力头数
NUM_LAYERS = 2  # Transformer编码器层数
VALIDATION_SIZE = 200  # 验证集样本数
FILL_VALUE = 2097  # 材料索引和厚度序列的填充值


# -------------------- 数据加载 --------------------
def load_tensor_data(tensor_path):
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"材料张量文件不存在: {tensor_path}")
    tensor_data = torch.load(tensor_path, weights_only=True)
    print(f"加载materials_tensor-->{tensor_data.shape}")
    return tensor_data

data_path = os.path.join(script_dir, 'train_data', 'train_data.pt')
data = load_tensor_data(data_path).to(device)
print(f"Training material_tensor shape: {data.shape}")

# 推断 max_layers
D = data.shape[1]
max_layers_plus1 = (D - 3) // 2
max_layers = max_layers_plus1 - 1
print(f"max_layers: {max_layers}, sequence length: {max_layers_plus1}")

# 加载材料张量并构建 mat_tensor (在1,3,5,10,14微米处的NK值)
tensor_path = os.path.join(script_dir, 'material_tensor', 'material_tensor.pt')
material_tensor = load_tensor_data(tensor_path).to(device)  # [num_materials, 1501, 2]
num_materials = material_tensor.shape[0]
indices = [50, 100, 300, 500, 1000]  # 对应1,3,5,10,14微米
mat_tensor = material_tensor[:, indices, :]  # [num_materials, 5, 2]
print(f"Material tensor shape: {material_tensor.shape}, mat_tensor shape: {mat_tensor.shape}")

# -------------------- 数据集划分 --------------------
total_samples = data.shape[0]
indices = list(range(total_samples))
random.shuffle(indices)
val_indices = indices[:VALIDATION_SIZE]
train_indices = indices[VALIDATION_SIZE:]

train_data = data[train_indices]
val_data = data[val_indices]
print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")


# -------------------- 自定义Dataset --------------------
class MultilayerDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
        self.start = data[:, 0].float()
        self.end = data[:, 1].float()
        self.material_indices = data[:, 2:2 + self.max_len].long()
        thickness_start = 2 + self.max_len
        self.thickness = data[:, thickness_start:thickness_start + self.max_len].float()
        self.fitness = data[:, -1].float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'start': self.start[idx],
            'end': self.end[idx],
            'material_indices': self.material_indices[idx],
            'thickness': self.thickness[idx],
            'fitness': self.fitness[idx]
        }


train_dataset = MultilayerDataset(train_data, max_layers_plus1)
val_dataset = MultilayerDataset(val_data, max_layers_plus1)


# -------------------- collate_fn: 动态构建材料特征 --------------------
def collate_fn(batch):
    start = torch.stack([item['start'] for item in batch])
    end = torch.stack([item['end'] for item in batch])
    material_indices = torch.stack([item['material_indices'] for item in batch])
    thickness = torch.stack([item['thickness'] for item in batch])
    fitness = torch.stack([item['fitness'] for item in batch])

    # 将厚度中的填充值替换为 0
    thickness = torch.where(material_indices != FILL_VALUE, thickness, torch.zeros_like(thickness))

    valid_layers = (material_indices != FILL_VALUE).sum(dim=1).float()

    # 构建材料特征（原有逻辑，注意 pad_value 可能有风险，但保持原样）
    pad_value = torch.zeros_like(material_indices)
    pad_value[:, 0] = 1
    safe_indices = torch.where(material_indices == FILL_VALUE, pad_value, material_indices)
    all_features = mat_tensor[safe_indices]  # [batch, seq_len, 5, 2]
    mask = (material_indices != FILL_VALUE).unsqueeze(-1).unsqueeze(-1)
    mat_features = torch.where(mask, all_features, torch.zeros_like(all_features))

    return {
        'start': start,
        'end': end,
        'material_indices': material_indices,
        'thickness': thickness,  # 此时填充位置已为0
        'fitness': fitness,
        'valid_layers': valid_layers,
        'mat_features': mat_features
    }


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# -------------------- 模型定义 --------------------
class MaterialCNN(nn.Module):
    """对每个层的5x2 NK数据进行CNN特征提取"""

    def __init__(self, in_channels=1, out_dim=D_CNN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 2), padding=(1, 0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, 5, 2]
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)
        features = self.conv(x)  # [batch*seq_len, out_dim]
        features = features.view(batch_size, seq_len, -1)
        return features


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x)
        return x


class ThicknessPredictor(nn.Module):
    def __init__(self, max_len, d_cnn=D_CNN, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.max_len = max_len
        self.cnn = MaterialCNN(out_dim=d_cnn)
        self.scalar_fc = nn.Linear(3, d_cnn)

        # CNN分支（多尺度）
        self.conv1d_low = nn.Conv1d(d_cnn, 64, kernel_size=3, padding=1)
        self.conv1d_high = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(max_len)
        )
        self.cnn_proj = nn.Linear(128, d_model)

        # Transformer分支
        self.transformer = TransformerBlock(d_model, nhead, num_layers, max_len)

        # 融合后预测厚度
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # 保证厚度非负
        )

    def forward(self, start, end, valid_layers, mat_features):
        cnn_out = self.cnn(mat_features)  # [batch, seq_len, d_cnn]

        scalar = torch.stack([start, end, valid_layers], dim=1)
        scalar_emb = self.scalar_fc(scalar).unsqueeze(1)  # [batch, 1, d_cnn]
        cnn_out = cnn_out + scalar_emb

        # CNN分支
        cnn_seq = cnn_out.permute(0, 2, 1)  # [batch, d_cnn, seq_len]
        low = F.relu(self.conv1d_low(cnn_seq))
        high = self.conv1d_high(low)
        high = high.permute(0, 2, 1)  # [batch, seq_len, 128]
        cnn_features = self.cnn_proj(high)  # [batch, seq_len, d_model]

        # Transformer分支
        trans_out = self.transformer(cnn_out)  # [batch, seq_len, d_model]

        # 融合
        fused = torch.cat([cnn_features, trans_out], dim=-1)  # [batch, seq_len, d_model*2]
        thickness = self.fc_out(fused).squeeze(-1)  # [batch, seq_len]
        return thickness


class FitnessPredictor(nn.Module):
    def __init__(self, max_len, d_cnn=D_CNN, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.max_len = max_len
        self.cnn = MaterialCNN(out_dim=d_cnn)
        self.scalar_fc = nn.Linear(3, d_cnn)
        self.thickness_fc = nn.Linear(1, d_cnn)

        # CNN分支
        self.conv1d_low = nn.Conv1d(d_cnn, 64, kernel_size=3, padding=1)
        self.conv1d_high = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(max_len)
        )
        self.cnn_proj = nn.Linear(128, d_model)

        # Transformer分支
        self.transformer = TransformerBlock(d_model, nhead, num_layers, max_len)

        # 融合后预测适应度
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, start, end, valid_layers, mat_features, thickness):
        # thickness 已经过掩码处理（填充位置为0）
        cnn_out = self.cnn(mat_features)  # [batch, seq_len, d_cnn]
        thickness_emb = self.thickness_fc(thickness.unsqueeze(-1))  # [batch, seq_len, d_cnn]
        combined = cnn_out + thickness_emb

        scalar = torch.stack([start, end, valid_layers], dim=1)
        scalar_emb = self.scalar_fc(scalar).unsqueeze(1)
        combined = combined + scalar_emb

        # CNN分支
        cnn_seq = combined.permute(0, 2, 1)  # [batch, d_cnn, seq_len]
        low = F.relu(self.conv1d_low(cnn_seq))
        high = self.conv1d_high(low)
        high = high.permute(0, 2, 1)
        cnn_features = self.cnn_proj(high)

        # Transformer分支
        trans_out = self.transformer(combined)

        # 融合
        fused = torch.cat([cnn_features, trans_out], dim=-1)  # [batch, seq_len, d_model*2]
        pooled = fused.mean(dim=1)  # [batch, d_model*2]
        fitness = self.fc_out(pooled).squeeze(-1)
        return fitness


class MultilayerNet(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.thickness_net = ThicknessPredictor(max_len)
        self.fitness_net = FitnessPredictor(max_len)

    def forward(self, start, end, valid_layers, mat_features, material_indices):
        pred_thickness = self.thickness_net(start, end, valid_layers, mat_features)
        # 创建厚度掩码：有效位置保留原值，填充位置置0
        thickness_mask = (material_indices != FILL_VALUE).float()  # [batch, seq_len]
        masked_thickness = pred_thickness * thickness_mask
        pred_fitness = self.fitness_net(start, end, valid_layers, mat_features, masked_thickness)
        return pred_thickness, pred_fitness


# -------------------- 初始化模型 --------------------
model = MultilayerNet(max_layers_plus1).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------- 加载已训练模型（如果存在） --------------------
model_path = os.path.join(script_dir, 'history_model', 'final_model.pth')
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"\nLoaded model from {model_path}")
else:
    print(f"\nNo pre-trained model found at {model_path}. Please train first.")
    # 如果不想训练，可以在这里退出
    exit()


# -------------------- 评估指标函数 --------------------
def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return mae, mse, r2


# -------------------- 验证并保存结果（排序对比 + 随机对比） --------------------
print("\nPerforming validation...")

# 提取最后100个样本作为验证集
val_data_tensor = data[-VALIDATION_SIZE:, :].clone()  # shape: [100, total_features]
true_fitness = val_data_tensor[:, -1].float()

# 按真实适应度从小到大排序
sorted_indices = torch.argsort(true_fitness)
sorted_val_data = val_data_tensor[sorted_indices]
sorted_true_fitness = true_fitness[sorted_indices].cpu().numpy()


# 定义输入预处理函数
def prepare_input(sample):
    start = sample[0].float().unsqueeze(0)
    end = sample[1].float().unsqueeze(0)
    material_indices = sample[2:2 + max_layers_plus1].long().unsqueeze(0)
    thickness = sample[2 + max_layers_plus1:2 + 2 * max_layers_plus1].float().unsqueeze(0)
    valid_layers = (material_indices != FILL_VALUE).sum(dim=1).float()

    pad_value = torch.zeros_like(material_indices)
    pad_value[:, 0] = 1
    safe_indices = torch.where(material_indices == FILL_VALUE, pad_value, material_indices)
    all_features = mat_tensor[safe_indices]
    mask = (material_indices != FILL_VALUE).unsqueeze(-1).unsqueeze(-1)
    mat_features = torch.where(mask, all_features, torch.zeros_like(all_features))

    return start.to(device), end.to(device), valid_layers.to(device), mat_features.to(device), material_indices.to(
        device)


model.eval()

# ---------- 图1：验证集排序后的真实值与预测值 ----------
pred_fitness_sorted = []
with torch.no_grad():
    for i in range(len(sorted_val_data)):
        sample = sorted_val_data[i]
        start, end, valid_layers, mat_features, material_indices = prepare_input(sample)
        _, pred = model(start, end, valid_layers, mat_features, material_indices)
        pred_fitness_sorted.append(pred.item())
pred_fitness_sorted = np.array(pred_fitness_sorted)

# 计算 Spearman 相关系数
from scipy.stats import spearmanr

spearman_corr, p_value = spearmanr(sorted_true_fitness, pred_fitness_sorted)
print(f"Spearman correlation (sorted validation): {spearman_corr:.4f} (p={p_value:.2e})")

# 绘制图1
plt.figure(figsize=(8, 5))
x = np.arange(1, len(sorted_true_fitness) + 1)
plt.plot(x, sorted_true_fitness, 'b-o', label='True Fitness', markersize=4, linewidth=1.5)
plt.plot(x, pred_fitness_sorted, 'r--s', label='Predicted Fitness', markersize=4, linewidth=1.5)
plt.xlabel('Sample Index (sorted by true fitness)')
plt.ylabel('Fitness')
plt.title('Validation Set: True vs Predicted Fitness (Sorted)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_validation_sorted.png', dpi=150)
plt.show()
print("Figure 1 saved as 'fig_validation_sorted.png'")

# ---------- 保存数据到 Excel ----------
# 保存排序后的验证数据
sorted_material_indices = sorted_val_data[:, 2:2 + max_layers_plus1].long().cpu().numpy()
sorted_thickness = sorted_val_data[:, 2 + max_layers_plus1:2 + 2 * max_layers_plus1].float().cpu().numpy()
sorted_material_str = [str(list(row)) for row in sorted_material_indices]
sorted_thickness_str = [str(list(row)) for row in sorted_thickness]

df_sorted = pd.DataFrame({
    'sample_index': x,
    'material_indices': sorted_material_str,
    'thickness': sorted_thickness_str,
    'true_fitness': sorted_true_fitness,
    'pred_fitness': pred_fitness_sorted
})
df_sorted.to_csv('validation_sorted.csv', index=False)
print("Sorted results saved to 'validation_sorted.csv'")

# ==================== 新增：保存所有验证样本的原始数据，并按预测值排序分析 ====================
print("\n--- Additional analysis: raw validation set and sorting by predicted fitness ---")

# 对原始验证集（未排序）进行预测，得到原始顺序的预测值
pred_fitness_original = []
with torch.no_grad():
    for i in range(len(val_data_tensor)):
        sample = val_data_tensor[i]
        start, end, valid_layers, mat_features, material_indices = prepare_input(sample)
        _, pred = model(start, end, valid_layers, mat_features, material_indices)
        pred_fitness_original.append(pred.item())
pred_fitness_original = np.array(pred_fitness_original)

# 原始真实值（未排序）
true_fitness_original = val_data_tensor[:, -1].float().cpu().numpy()

# 保存原始数据到Excel
raw_material_indices = val_data_tensor[:, 2:2 + max_layers_plus1].long().cpu().numpy()
raw_thickness = val_data_tensor[:, 2 + max_layers_plus1:2 + 2 * max_layers_plus1].float().cpu().numpy()
raw_material_str = [str(list(row)) for row in raw_material_indices]
raw_thickness_str = [str(list(row)) for row in raw_thickness]

df_raw = pd.DataFrame({
    'sample_index': np.arange(1, len(true_fitness_original) + 1),
    'material_indices': raw_material_str,
    'thickness': raw_thickness_str,
    'true_fitness': true_fitness_original,
    'pred_fitness': pred_fitness_original
})
df_raw.to_csv('validation_raw_all.csv', index=False)
print("Raw validation material_tensor saved to 'validation_raw_all.csv'")

# 计算原始顺序的MSE（与排序无关，仅作为参考）
mae_original = np.mean(np.abs(true_fitness_original - pred_fitness_original))

mse_original = np.mean((true_fitness_original - pred_fitness_original) ** 2)

ss_res = np.sum((true_fitness_original - pred_fitness_original) ** 2)
ss_tot = np.sum((true_fitness_original - np.mean(true_fitness_original)) ** 2)
r2_original = 1 - (ss_res / ss_tot)
print(f"Validation material_tensor: MAE={mae_original:.6f}, "
      f"                 MSE={mse_original:.6f}"
      f"                 R^2={r2_original:.6f}")

# 计算Spearman秩相关系数（衡量排序一致性）
from scipy.stats import spearmanr

spearman_overall, p_overall = spearmanr(true_fitness_original, pred_fitness_original)
print(f"Spearman correlation (overall): {spearman_overall:.4f} (p={p_overall:.2e})")

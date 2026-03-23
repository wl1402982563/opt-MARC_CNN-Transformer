import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import pandas as pd
import os
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=FutureWarning)


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

# -------------------- 超参数（与训练时一致） --------------------
BATCH_SIZE = 500
EPOCHS = 200
LEARNING_RATE = 1e-3
LAMBDA_THICKNESS = 0.2
LAMBDA_FITNESS = 1
D_CNN = 512
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 2
VALIDATION_SIZE = 500
FILL_VALUE = 2097


# -------------------- 模型定义（与训练时完全一致） --------------------
class MaterialCNN(nn.Module):
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
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)
        features = self.conv(x)
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
        self.conv1d_low = nn.Conv1d(d_cnn, 64, kernel_size=3, padding=1)
        self.conv1d_high = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(max_len)
        )
        self.cnn_proj = nn.Linear(128, d_model)
        self.transformer = TransformerBlock(d_model, nhead, num_layers, max_len)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )

    def forward(self, start, end, valid_layers, mat_features):
        cnn_out = self.cnn(mat_features)
        scalar = torch.stack([start, end, valid_layers], dim=1)
        scalar_emb = self.scalar_fc(scalar).unsqueeze(1)
        cnn_out = cnn_out + scalar_emb
        cnn_seq = cnn_out.permute(0, 2, 1)
        low = F.relu(self.conv1d_low(cnn_seq))
        high = self.conv1d_high(low)
        high = high.permute(0, 2, 1)
        cnn_features = self.cnn_proj(high)
        trans_out = self.transformer(cnn_out)
        fused = torch.cat([cnn_features, trans_out], dim=-1)
        thickness = self.fc_out(fused).squeeze(-1)
        return thickness


class FitnessPredictor(nn.Module):
    def __init__(self, max_len, d_cnn=D_CNN, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.max_len = max_len
        self.cnn = MaterialCNN(out_dim=d_cnn)
        self.scalar_fc = nn.Linear(3, d_cnn)
        self.thickness_fc = nn.Linear(1, d_cnn)
        self.conv1d_low = nn.Conv1d(d_cnn, 64, kernel_size=3, padding=1)
        self.conv1d_high = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(max_len)
        )
        self.cnn_proj = nn.Linear(128, d_model)
        self.transformer = TransformerBlock(d_model, nhead, num_layers, max_len)
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
        cnn_out = self.cnn(mat_features)
        thickness_emb = self.thickness_fc(thickness.unsqueeze(-1))
        combined = cnn_out + thickness_emb
        scalar = torch.stack([start, end, valid_layers], dim=1)
        scalar_emb = self.scalar_fc(scalar).unsqueeze(1)
        combined = combined + scalar_emb
        cnn_seq = combined.permute(0, 2, 1)
        low = F.relu(self.conv1d_low(cnn_seq))
        high = self.conv1d_high(low)
        high = high.permute(0, 2, 1)
        cnn_features = self.cnn_proj(high)
        trans_out = self.transformer(combined)
        fused = torch.cat([cnn_features, trans_out], dim=-1)
        pooled = fused.mean(dim=1)
        fitness = self.fc_out(pooled).squeeze(-1)
        return fitness


class MultilayerNet(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.thickness_net = ThicknessPredictor(max_len)
        self.fitness_net = FitnessPredictor(max_len)

    def forward(self, start, end, valid_layers, mat_features, material_indices):
        pred_thickness = self.thickness_net(start, end, valid_layers, mat_features)
        thickness_mask = (material_indices != FILL_VALUE).float()
        masked_thickness = pred_thickness * thickness_mask
        pred_fitness = self.fitness_net(start, end, valid_layers, mat_features, masked_thickness)
        return pred_thickness, pred_fitness


# -------------------- 加载训练好的模型 --------------------
model_path = os.path.join('history_model', 'final_model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

checkpoint = torch.load(model_path, map_location=device)

temp_data = torch.load('train_data_percent/train_data_10percent.pt', weights_only=True)
D = temp_data.shape[1]
max_layers_plus1 = (D - 3) // 2
max_layers = max_layers_plus1 - 1
print(f"Inferred max_layers_plus1: {max_layers_plus1}")

model = MultilayerNet(max_layers_plus1).to(device)
model.load_state_dict(checkpoint)
model.eval()
print(f"Loaded model from {model_path}")


# -------------------- 定义输入预处理函数（与之前相同，但使用动态的 mat_tensor） --------------------
def prepare_input(sample, mat_tensor):
    start = sample[0].float().unsqueeze(0)
    end = sample[1].float().unsqueeze(0)
    material_indices = sample[2:2 + max_layers_plus1].long().unsqueeze(0)
    thickness = sample[2 + max_layers_plus1:2 + 2 * max_layers_plus1].float().unsqueeze(0)
    valid_layers = (material_indices != FILL_VALUE).sum(dim=1).float()

    pad_value = torch.zeros_like(material_indices)
    pad_value[:, 0] = 1
    safe_indices = torch.where(material_indices == FILL_VALUE, pad_value, material_indices)
    all_features = mat_tensor[safe_indices]  # 使用传入的 mat_tensor
    mask = (material_indices != FILL_VALUE).unsqueeze(-1).unsqueeze(-1)
    mat_features = torch.where(mask, all_features, torch.zeros_like(all_features))

    return start.to(device), end.to(device), valid_layers.to(device), mat_features.to(device), material_indices.to(
        device)


# -------------------- 定义波长点索引（与训练时一致） --------------------
indices = [50, 100, 300, 500, 1000]  # 对应 1,3,5,10,14 微米

# -------------------- 百分比列表 --------------------
percent_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# -------------------- 结果存储 --------------------
results = []

for percent in percent_list:
    print(f"\n========== Processing {percent}% new materials ==========")
    # 加载数据文件
    data_file = f'train_data_percent/train_data_{percent}percent.pt'
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found, skipping")
        continue
    val_data = torch.load(data_file, weights_only=True).to(device)
    print(f"Loaded data shape: {val_data.shape}")

    # 加载对应的新材料张量
    mat_file = f'material_tensor/new_materials_{percent}percent.pt'
    if not os.path.exists(mat_file):
        print(f"Warning: {mat_file} not found, skipping")
        continue

    new_mat_tensor = torch.load(mat_file, weights_only=True).to(device)  # [num_materials, 1501, 2]
    # 提取5个波长点的 nk 值
    mat_tensor = new_mat_tensor[:, indices, :]  # [num_materials, 5, 2]
    print(f"Material tensor shape for this percent: {mat_tensor.shape}")

    # 提取真实适应度（最后一列）
    true_fitness = val_data[:, -1].float().cpu().numpy()

    # 预测适应度
    pred_fitness = []
    with torch.no_grad():
        for i in range(len(val_data)):
            sample = val_data[i]
            # print("sample = ", sample)
            start, end, valid_layers, mat_features, material_indices = prepare_input(sample, mat_tensor)
            _, pred = model(start, end, valid_layers, mat_features, material_indices)
            pred_fitness.append(pred.item())
    pred_fitness = np.array(pred_fitness)

    # 计算 MSE
    mse = np.mean((true_fitness - pred_fitness) ** 2)

    # 计算 Spearman 相关系数
    spearman_corr, p_value = spearmanr(true_fitness, pred_fitness)

    print(f"Results: MSE = {mse:.6f}, Spearman = {spearman_corr:.4f} (p={p_value:.2e})")

    results.append({
        'percent': percent,
        'mse': mse,
        'spearman': spearman_corr
    })

# -------------------- 保存结果到 CSV --------------------
df_results = pd.DataFrame(results)
df_results.to_csv('validation_percent_results.csv', index=False)
print("\nResults saved to 'validation_percent_results.csv'")

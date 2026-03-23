import pandas as pd
import os

import torch
import pickle
# 材料列表
materials = ("ZnSe", "ZnS", "SiO2", "Al2O3", "InAs", "GaAs",
             "Ta2O5", "Si3N4", "TiO2", "ZrO2", "Glass", "GaSb")

# 存放Excel文件的目录（请修改为您的实际路径）
base_directory = r"G:\Good_Good_Study\Python_wl\MARCs\MARCs_GRU+Self-Attention\材料库"

# 存储结果的字典
wavelength_ranges = {}

for material in materials:
    file_path = os.path.join(base_directory, f"{material}.xlsx")
    if not os.path.exists(file_path):
        print(f"⚠ 文件不存在: {file_path}")
        wavelength_ranges[material] = (None, None)
        continue

    try:
        df = pd.read_excel(file_path)
        # 检查列是否存在
        if 'L' not in df.columns:
            print(f"⚠ 文件 {material}.xlsx 中缺少 'L' 列")
            wavelength_ranges[material] = (None, None)
            continue

        wavelengths = df['L'].dropna().values
        if len(wavelengths) == 0:
            print(f"⚠ 文件 {material}.xlsx 中波长数据为空")
            wavelength_ranges[material] = (None, None)
            continue

        min_wl = wavelengths.min()
        max_wl = wavelengths.max()
        wavelength_ranges[material] = (min_wl, max_wl)
        print(f"√ {material}: 波长范围 {min_wl:.4f} – {max_wl:.4f} μm")

    except Exception as e:
        print(f"✗ 读取 {material}.xlsx 时出错: {e}")
        wavelength_ranges[material] = (None, None)

# 汇总输出
print("\n" + "="*50)
# print("各材料波长范围统计：")
# for material, (min_wl, max_wl) in wavelength_ranges.items():
#     if min_wl is not None:
#         print(f"{material:8s}: {min_wl:.4f} – {max_wl:.4f} μm")
#     else:
#         print(f"{material:8s}: 数据缺失")
#

# 加载材料名称
with open('sorted_material_names.pkl', 'rb') as f:
    material_names = pickle.load(f)   # 列表，顺序与 material_tensor 第一维对应

# 加载材料张量
material_tensor = torch.load('material_tensor.pt', weights_only=True)  # shape: [14, 1501, 2]

# 定义目标波长 (μm) 及其在张量中的索引（需根据您的波长数组确定）
target_wavelengths = [0.5, 1, 3, 5, 10]
# 假设 wavelength 数组已定义为线性或已知，请根据实际设置索引，例如：
wavelengths = torch.linspace(0, 15, 1501)  # 仅为示例
indices = [torch.argmin(torch.abs(wavelengths - w)).item() for w in target_wavelengths]

# 提取数据
data = {}   # 字典，键为材料名，值为 (n_list, k_list)
for i, name in enumerate(material_names):
    nk = material_tensor[i, indices, :]   # [5, 2]
    n_vals = nk[:, 0].tolist()
    k_vals = nk[:, 1].tolist()
    data[name] = (n_vals, k_vals)

# 打印 LaTeX 表格行（可用于后续填充）
for name in material_names:
    n_vals, k_vals = data[name]
    n_str = ", ".join([f"{v:.4f}" for v in n_vals])
    k_str = ", ".join([f"{v:.4f}" for v in k_vals])
    print(f"{name} & {n_str} & {k_str} \\\\")
import torch
import pickle

# 加载材料名称
with open('sorted_material_names.pkl', 'rb') as f:
    material_names = pickle.load(f)  # 列表，顺序与 material_tensor 第一维对应

# 加载材料张量
material_tensor = torch.load('material_tensor.pt', weights_only=True)  # shape: [14, 1501, 2]
print("\n\n", ">" * 20, f"共{material_tensor.shape[0]}种材料", "<" * 20, "\n")
# 定义目标波长 (μm) 及其在张量中的索引（需根据您的波长数组确定）
target_wavelengths = [0.5, 1, 3, 5, 10]
# 假设 wavelength 数组已定义为线性或已知，请根据实际设置索引，例如：
wavelengths = torch.linspace(0.1, 15, 1501)  # 仅为示例
indices = [torch.argmin(torch.abs(wavelengths - w)).item() for w in target_wavelengths]

# 提取数据
data = {}  # 字典，键为材料名，值为 (n_list, k_list)
for i, name in enumerate(material_names):
    nk = material_tensor[i, indices, :]  # [5, 2]
    n_vals = nk[:, 0].tolist()
    k_vals = nk[:, 1].tolist()
    data[name] = (n_vals, k_vals)

# 打印 LaTeX 表格行（可用于后续填充）
for name in material_names:
    n_vals, k_vals = data[name]
    n_str = ", ".join([f"{v:.4f}" for v in n_vals])
    k_str = ", ".join([f"{v:.4f}" for v in k_vals])
    print(f"{name} & {n_str} & {k_str}")

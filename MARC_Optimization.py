import os
import torch
from Network_Model import MultilayerNet, FILL_VALUE
from Material_Optimizer import MaterialDEOptimizer
from Adjustment_Thickness import ThicknessDEOptimizer
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from Calculate_ATR_Thickness import calculate_batch_rta, replace_material_with_nk_simple
import inspect
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

def to_torch_float32(data):
    if isinstance(data, torch.Tensor):
        return data.double().to(device)
    elif isinstance(data, (list, tuple, np.ndarray)):
        data = np.array(data, dtype=np.float32)
        tensor = torch.from_numpy(data)
    elif isinstance(data, (int, float)):
        tensor = torch.tensor([data], dtype=torch.float32)
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")
    return tensor.to(device)


def find_best_structure_and_plot(x, y1, y2, y3):
    plt.figure(figsize=(12, 8))
    x = x.cpu().detach().numpy()
    y1 = y1.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()
    y3 = y3.cpu().detach().numpy()
    # 打印统计信息
    print(f"R: mean={np.mean(y1):.3f}, Max={np.max(y1):.3f}")
    print(f"T: mean={np.mean(y2):.3f}, Max={np.max(y2):.3f}")
    print(f"A: mean={np.mean(y3):.3f}, Max={np.max(y3):.3f}")

    # 绘制曲线
    line1, = plt.plot(x, y1, color='#1f77b4', linewidth=2.5, label='Reflectivity(R)')
    line2, = plt.plot(x, y2, color='#2ca02c', linewidth=2.5, label='Transmissivity(T)')
    line3, = plt.plot(x, y3, color='#d62728', linewidth=2.5, label='Absorptivity(A)')

    # 设置标题和标签
    plt.xlabel('Wavelength (μm)', fontsize=14, labelpad=10)
    plt.ylabel('MARC performance', fontsize=14, labelpad=10)

    # 设置坐标轴范围
    plt.ylim(0.0, 1.05)

    # 添加图例
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加物理约束线 (R+T+A=1)
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='R+T+A=1')

    # 美化坐标轴
    plt.tick_params(axis='both', which='major', labelsize=11)

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
    return


# -------------------- 加载材料张量和模型 --------------------
name_path = os.path.join(script_dir, 'material_tensor', 'sorted_material_names.pkl')
tensor_path = os.path.join(script_dir, 'material_tensor', 'material_tensor.pt')
material_tensor = torch.load(tensor_path, weights_only=True).to(device)

num_materials = material_tensor.shape[0]
indices = [50, 100, 300, 500, 1000]  # 必须与训练时一致
mat_tensor = material_tensor[:, indices, :]  # [num_materials, 5, 2]
mat_tensor = mat_tensor.to(device)

model = MultilayerNet().to(device)
model_path = os.path.join(script_dir, 'history_model', 'final_model.pth')
checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
print("\nCNN+T Model loaded.")


# -------------------- 求解目标 --------------------
def merge_duplicate_materials(material_seq, thickness_seq):
    # 将可能的torch张量转换为Python标量
    material_seq = [int(m.cpu()) if torch.is_tensor(m) else m for m in material_seq]
    thickness_seq = [float(t.cpu()) if torch.is_tensor(t) else t for t in thickness_seq]

    material_dict = {}
    for mat, thick in zip(material_seq, thickness_seq):
        if mat in material_dict:
            material_dict[mat] += thick
        else:
            material_dict[mat] = thick

    new_materials = []
    new_thickness = []
    for mat in material_seq:
        if mat not in new_materials:
            new_materials.append(mat)
            new_thickness.append(material_dict[mat])
    return new_materials, new_thickness


with open(name_path, 'rb') as f:
    material_names = pickle.load(f)
    
start_wl = 2
end_wl = 16
substrate_idx = material_names.index('ZnSe')  # 基底材料索引，假设为 5
layers = [2, 4, 8, 10, 14]

# 初始化最佳记录
best_overall_fitness = -float('inf')
best_overall_material = None
best_overall_thickness = None
best_overall_layers = None

for num_layers in layers:
    single_time = time.time()
    # -------------------- 材料优化 --------------------
    print(f"\nStarting {num_layers} materials optimization...")
    material_opt = MaterialDEOptimizer(
        model=model,
        mat_tensor=mat_tensor,
        start_wl=start_wl,
        end_wl=end_wl,
        substrate_idx=substrate_idx,
        num_layers=num_layers,
        max_len=15,
        pop_size=200,
        max_generations=100
    )
    best_material_seq, best_material_fitness = material_opt.optimize()
    print(f"Best material sequence(idx): {best_material_seq[: num_layers].cpu().tolist()}")
    # print(f"Best fitness from NN: {best_material_fitness:.6f}")

    # 提取有效材料序列（前 num_layers 层）
    effective_material_seq = best_material_seq[:num_layers].clone()

    # -------------------- 厚度优化 --------------------
    print("Starting thickness optimization...")
    thickness_opt = ThicknessDEOptimizer(
        material_tensor=material_tensor,
        start_wl=start_wl,
        end_wl=end_wl,
        material_seq=effective_material_seq,
        num_layers=num_layers,
        pop_size=300,
        max_generations=300
    )
    best_thickness, best_thickness_fitness = thickness_opt.optimize()
    print(f"Best fitness from physics: {best_thickness_fitness:.6f}")

    # 更新全局最佳
    if best_thickness_fitness > best_overall_fitness:
        best_overall_fitness = best_thickness_fitness
        best_overall_material = effective_material_seq.clone()
        best_overall_thickness = best_thickness.clone()
        best_overall_layers = num_layers
    print(f"Single time taken: {time.time() - single_time:.4f}s")
# -------------------- 最终结果 --------------------

print("\n=== Final Results ===")
new_material, new_thickness = merge_duplicate_materials(best_overall_material, best_overall_thickness)
material_name_list = [material_names[idx] for idx in new_material]
print(f"Material sequence: {material_name_list}")
print(f"Thickness sequence: {new_thickness}")
print(f"Best fitness: {best_overall_fitness:.6f}")
print(f"Total time taken: {time.time() - start_time:.6f}s")

# 查看光谱
wavelengths = [start_wl, end_wl]
wavelengths = to_torch_float32(wavelengths)
material_seq = to_torch_float32(new_material)
thickness_seq = to_torch_float32(new_thickness)

wavelengths_calc, materials_nk, thicknesses = replace_material_with_nk_simple(
    wavelengths, material_seq.unsqueeze(0), thickness_seq.unsqueeze(0).unsqueeze(0), material_tensor
)

R, T, A = calculate_batch_rta(wavelengths_calc, materials_nk, thicknesses)

lambda0, lambda1 = wavelengths[0].item(), wavelengths[1].item()
wavelength = torch.arange(
        lambda0,
        lambda1,
        max(0.05, round((lambda1 - lambda0) / 70 * 100) / 100)
    )
find_best_structure_and_plot(wavelength,
                             R.squeeze(0).squeeze(0),
                             T.squeeze(0).squeeze(0),
                             A.squeeze(0).squeeze(0))

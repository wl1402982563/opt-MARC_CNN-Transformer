import torch
import numpy as np
from Network_Model import MultilayerNet, FILL_VALUE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaterialDEOptimizer:
    def __init__(self, model, mat_tensor, start_wl, end_wl, substrate_idx,
                 num_layers=6, max_len=15, pop_size=50, max_generations=100,
                 crossover_rate=0.7, mutation_factor=0.8):

        self.model = model
        self.mat_tensor = mat_tensor.to(device)
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.substrate_idx = substrate_idx
        self.num_layers = num_layers          # 有效层数（包括基底）
        self.max_len = max_len                 # 模型输入需要的完整长度（含填充）
        self.fill_value = FILL_VALUE
        self.num_materials = mat_tensor.shape[0]

        # DE参数
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor

        # 种群管理：存储有效层序列 [pop_size, num_layers]
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.best_individual_history = []

        # 可用材料列表（排除基底）
        self.candidate_values = torch.tensor(
            [i for i in range(self.num_materials) if i != self.substrate_idx],
            device=device
        )

        # 初始化种群
        self.init_population()

    def _pad_population(self, population):
        """
        将有效层序列 population [batch, num_layers] 补齐到 [batch, max_len]
        填充位置用 fill_value
        """
        batch_size = population.shape[0]
        padded = torch.full((batch_size, self.max_len), self.fill_value, dtype=torch.long, device=device)
        padded[:, :self.num_layers] = population
        return padded

    def _prepare_input(self, padded_population):
        """padded_population: [batch, max_len] 已包含填充值"""
        batch_size = padded_population.shape[0]
        start = torch.full((batch_size,), self.start_wl, dtype=torch.float32, device=device)
        end = torch.full((batch_size,), self.end_wl, dtype=torch.float32, device=device)
        valid_layers = (padded_population != self.fill_value).sum(dim=1).float()

        # 构建 mat_features
        pad_value = torch.zeros_like(padded_population)
        pad_value[:, 0] = 1
        safe_indices = torch.where(padded_population == self.fill_value, pad_value, padded_population)
        all_features = self.mat_tensor[safe_indices]  # [batch, max_len, 5, 2]
        mask = (padded_population != self.fill_value).unsqueeze(-1).unsqueeze(-1)
        mat_features = torch.where(mask, all_features, torch.zeros_like(all_features))

        return start, end, valid_layers, mat_features, padded_population

    def evaluate_population(self, population):
        """
        population: [batch, num_layers] 有效层序列
        返回排序后的适应度和对应的有效层序列
        """
        padded = self._pad_population(population)
        start, end, valid_layers, mat_features, padded_pop = self._prepare_input(padded)

        with torch.no_grad():
            _, fitness = self.model(start, end, valid_layers, mat_features, padded_pop)

        # 按适应度降序排序
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        sorted_population = population[sorted_indices]
        return sorted_fitness, sorted_population

    def vectorized_replace(self, new_fitness, new_population, fitness, population):

        Disturbance_pop_size = 500
        indices = torch.randint(0, len(self.candidate_values), (Disturbance_pop_size, self.num_layers - 1), device=device)
        front_layers = self.candidate_values[indices]
        substrate_layer = torch.full((Disturbance_pop_size, 1), self.substrate_idx, device=device)
        effective_layers = torch.cat([front_layers, substrate_layer], dim=1)
        fitness_new, population_new = self.evaluate_population(effective_layers)

        all_fitness = torch.cat([fitness, new_fitness, fitness_new], dim=0)
        all_population = torch.cat([population, new_population, population_new], dim=0)

        sorted_all, all_indices = torch.sort(all_fitness, descending=True)
        sorted_population = all_population[all_indices]

        argmax_fitness = sorted_all[:len(fitness)]
        argmax_population = sorted_population[:len(fitness)]
        return argmax_fitness, argmax_population

    def init_population(self):
        # 前 num_layers-1 层从候选材料中随机选择
        indices = torch.randint(0, len(self.candidate_values), (self.pop_size, self.num_layers - 1), device=device)
        front_layers = self.candidate_values[indices]  # [pop_size, num_layers-1]

        # 基底层
        substrate_layer = torch.full((self.pop_size, 1), self.substrate_idx, device=device)

        # 合并得到完整有效层 [pop_size, num_layers]
        effective_layers = torch.cat([front_layers, substrate_layer], dim=1)

        # 评估种群，得到排序后的种群和适应度
        self.fitness, self.population = self.evaluate_population(effective_layers)

    def differential_mutation(self):
        """
        对当前种群（有效层）进行差分变异，返回试验种群（有效层）
        """
        pop_size = self.population.shape[0]
        # 为每个个体选择三个不同的随机索引
        indices = torch.randint(0, pop_size - 1, (pop_size, 3), device=device)

        positions = torch.arange(pop_size, device=device).unsqueeze(1)
        indices = torch.where(indices >= positions, indices + 1, indices)
        indices = indices % pop_size

        r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]

        # 克隆当前种群作为供体基础
        donor = self.population[r1].clone()  # [pop_size, num_layers]

        # 对可变层（前 num_layers-1 层）进行变异
        delta = self.mutation_factor * (self.population[r2][:, :self.num_layers - 1].float() -
                                        self.population[r3][:, :self.num_layers - 1].float())
        donor[:, :self.num_layers - 1] = torch.round(
            self.population[r1][:, :self.num_layers - 1].float() + torch.abs(delta)
        ).long()

        # 确保基底层不变
        donor[:, self.num_layers - 1] = self.substrate_idx

        # 边界约束
        donor[:, :self.num_layers - 1] = torch.clamp(donor[:, :self.num_layers - 1], 0, self.num_materials - 1)

        return donor

    def crossover(self, donor):
        """
        对供体和当前种群进行二项式交叉，返回试验种群
        donor: [pop_size, num_layers] 供体（可变层已变异）
        """
        pop_size = self.population.shape[0]
        # 交叉掩码只对可变层
        cross_mask = torch.rand(pop_size, self.num_layers - 1, device=device) < self.crossover_rate

        # 确保至少有一个维度来自donor
        j_rand = torch.randint(0, self.num_layers - 1, (pop_size,), device=device)
        row_indices = torch.arange(pop_size, device=device)
        cross_mask[row_indices, j_rand] = True

        # 生成试验种群
        trial = self.population.clone()
        trial[:, :self.num_layers - 1] = torch.where(
            cross_mask,
            donor[:, :self.num_layers - 1],
            self.population[:, :self.num_layers - 1]
        )
        # 基底层不变
        trial[:, self.num_layers - 1] = self.substrate_idx

        # ========== 修复：确保可变层不含基底材料 ==========
        # 找出可变层中等于基底索引的位置
        mask_substrate = (trial[:, :self.num_layers - 1] == self.substrate_idx)
        if mask_substrate.any():
            # 为每个需要修复的位置重新随机采样非基底材料
            # 生成足够多的随机索引
            num_fix = mask_substrate.sum().item()
            random_indices = torch.randint(0, len(self.candidate_values), (num_fix,), device=device)
            random_materials = self.candidate_values[random_indices]
            # 将修复值填入对应位置
            trial[:, :self.num_layers - 1][mask_substrate] = random_materials

        return trial

    def selection(self, trial_population):
        """选择操作：评估试验种群，与当前种群合并保留更优"""
        trial_fitness, trial_population = self.evaluate_population(trial_population)
        self.fitness, self.population = self.vectorized_replace(
            trial_fitness, trial_population,
            self.fitness, self.population
        )

    def merge_and_select(self):
        """一次迭代：变异、交叉、选择"""
        donor = self.differential_mutation()
        trial = self.crossover(donor)
        self.selection(trial)

        # 记录最优
        best_idx = torch.argmax(self.fitness)
        best_fitness = self.fitness[best_idx].item()
        best_individual = self.population[best_idx].clone()
        self.best_fitness_history.append(best_fitness)
        self.best_individual_history.append(best_individual)

    def optimize(self):
        for gen in range(self.max_generations):
            self.merge_and_select()

        # 返回最优个体（有效层）和最优适应度
        best_idx = torch.argmax(self.fitness)
        best_individual = self.population[best_idx].clone()
        best_fitness = self.fitness[best_idx].item()
        return best_individual, best_fitness
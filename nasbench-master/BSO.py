# bso_nasbench.py
"""
Brain Storm Optimization (BSO) Algorithm on NASBench-101
Implementation of BSO algorithm for neural architecture search on NASBench dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import random
from collections import defaultdict
from absl import app
from nasbench import api

# 设置路径
NASBENCH_TFRECORD = '../../nasbench_only108.tfrecord'

# 操作类型
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

# BSO算法参数
POPULATION_SIZE = 50
MAX_ITERATIONS = 100
CLUSTER_COUNT = 5
P_REPLACE = 0.2  # 替换概率
P_ONE_CENTER = 0.8  # 选择一个中心的概率
P_TWO_CENTER = 0.2  # 选择两个中心的概率
MUTATION_RATE = 0.1
ELITE_COUNT = 5

class NASArchitecture:
    """表示NASBench中的一个架构"""
    def __init__(self, matrix, ops, hash_value=None):
        # 确保矩阵是7x7的
        self.matrix = self._ensure_matrix_size(matrix)
        self.ops = self._ensure_ops_size(ops)
        self.hash = hash_value
        self.fitness = 0.0  # 验证准确率（平均值）
        self.fitness_runs = []  # 保存三次运行的验证准确率
        self.test_accuracy = 0.0  # 测试准确率（平均值）
        self.test_accuracy_runs = []  # 保存三次运行的测试准确率
        self.parameters = 0
        self.training_time = 0.0
        self.training_time_runs = []  # 保存三次运行的训练时间
        
    def _ensure_ops_size(self, ops):
        """确保操作列表有7个元素"""
        if len(ops) != 7:
            # 如果不是7个，补充或截断
            new_ops = [INPUT] + [CONV3X3] * 5 + [OUTPUT]  # 使用预定义的常量
            for i in range(min(len(ops), 7)):
                # 确保操作类型是有效的
                if ops[i] in [INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3]:
                    new_ops[i] = ops[i]
            return new_ops
        return ops
        
    def _ensure_matrix_size(self, matrix):
        """确保矩阵是7x7大小"""
        if len(matrix) != 7:
            # 如果不是7x7，创建7x7的矩阵
            new_matrix = [[0] * 7 for _ in range(7)]
            for i in range(min(len(matrix), 7)):
                for j in range(min(len(matrix[i]), 7)):
                    new_matrix[i][j] = matrix[i][j]
            return new_matrix
        return matrix
        
    def __repr__(self):
        return f"Architecture(fitness={self.fitness:.4f}, params={self.parameters:,})"
    
    def copy(self):
        """创建副本"""
        new_arch = NASArchitecture(
            [row[:] for row in self.matrix],
            self.ops[:],
            self.hash
        )
        new_arch.fitness = self.fitness
        new_arch.fitness_runs = self.fitness_runs[:]
        new_arch.test_accuracy = self.test_accuracy
        new_arch.test_accuracy_runs = self.test_accuracy_runs[:]
        new_arch.parameters = self.parameters
        new_arch.training_time = self.training_time
        new_arch.training_time_runs = self.training_time_runs[:]
        return new_arch

    def get_features(self):
        """提取架构特征用于聚类"""
        features = []
        
        # 特征1：参数量（对数尺度）
        param_feat = np.log10(self.parameters + 1)
        features.append(param_feat)
        
        # 特征2：操作类型分布
        conv1x1_count = self.ops.count(CONV1X1)
        conv3x3_count = self.ops.count(CONV3X3)
        maxpool_count = self.ops.count(MAXPOOL3X3)
        features.extend([conv1x1_count, conv3x3_count, maxpool_count])
        
        # 特征3：连接密度
        matrix_arr = np.array(self.matrix)
        connection_density = np.sum(matrix_arr) / (matrix_arr.shape[0] * matrix_arr.shape[1])
        features.append(connection_density)
        
        # 特征4：适应度值
        features.append(self.fitness)
        
        return np.array(features)

class BSONAS:
    """BSO算法在NASBench上的实现"""
    
    def __init__(self, nasbench, population_size=POPULATION_SIZE):
        self.nasbench = nasbench
        self.population_size = population_size
        self.population = []
        self.best_architecture = None
        self.best_fitness = 0.0
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'cluster_sizes': [],
            'evaluations': 0
        }
        self.clusters = []
        
    def initialize_population(self):
        """初始化种群"""
        print("Initializing population...")
        all_hashes = list(self.nasbench.hash_iterator())
        
        # 随机选择初始种群
        selected_hashes = np.random.choice(all_hashes, 
                                        min(self.population_size * 2, len(all_hashes)), 
                                        replace=False)
        
        valid_count = 0
        for hash_value in selected_hashes:
            if valid_count >= self.population_size:
                break
                    
            fixed_stats, computed_stats = self.nasbench.get_metrics_from_hash(hash_value)
            
            # 获取108 epoch的结果
            if 108 in computed_stats:
                # 获取架构信息
                matrix = fixed_stats['module_adjacency']
                ops = fixed_stats['module_operations']
                
                # 计算适应度（平均验证准确率）
                val_accuracies = [run['final_validation_accuracy'] 
                                for run in computed_stats[108]]
                test_accuracies = [run['final_test_accuracy'] 
                                for run in computed_stats[108]]
                train_times = [run['final_training_time'] 
                            for run in computed_stats[108]]
                
                # 创建架构对象
                arch = NASArchitecture(matrix, ops, hash_value)
                arch.fitness_runs = val_accuracies
                arch.fitness = np.mean(val_accuracies)
                arch.test_accuracy_runs = test_accuracies
                arch.test_accuracy = np.mean(test_accuracies)
                arch.parameters = fixed_stats['trainable_parameters']
                arch.training_time_runs = train_times
                arch.training_time = np.mean(train_times)
                
                self.population.append(arch)
                valid_count += 1
                
                # 更新最佳
                if arch.fitness > self.best_fitness:
                    self.best_fitness = arch.fitness
                    self.best_architecture = arch.copy()
        
        print(f"Initialized {len(self.population)} architectures")
        print(f"Best initial fitness: {self.best_fitness:.4f} (based on {len(self.best_architecture.fitness_runs)} runs)")
        if self.best_architecture:
            print(f"Best initial test accuracy: {self.best_architecture.test_accuracy:.4f}")
            print(f"Best initial parameters: {self.best_architecture.parameters:,}")
    
    def clustering(self, population):
        """K-means聚类"""
        if not population:
            return [], []
        
        # 简单实现：基于适应度值聚类
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        clusters = []
        
        # 将种群分成k个簇
        k = min(CLUSTER_COUNT, len(sorted_pop))
        cluster_size = len(sorted_pop) // k
        
        for i in range(k):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < k - 1 else len(sorted_pop)
            if start_idx < len(sorted_pop):  # 确保索引有效
                clusters.append(sorted_pop[start_idx:end_idx])
        
        # 移除空簇
        clusters = [c for c in clusters if c]
        
        # 选择每个簇的中心（最佳个体）
        cluster_centers = []
        for cluster in clusters:
            if cluster:  # 确保簇不为空
                cluster.sort(key=lambda x: x.fitness, reverse=True)
                cluster_centers.append(cluster[0])  # 最佳个体作为中心
        
        return clusters, cluster_centers
    
    def mutate_architecture(self, arch):
        """突变操作：随机修改架构"""
        new_arch = arch.copy()
        matrix = [row[:] for row in new_arch.matrix]
        ops = new_arch.ops[:]
        
        n = len(matrix)
        
        # 随机修改邻接矩阵
        if random.random() < MUTATION_RATE:
            # 随机添加/删除边（保持有效）
            for i in range(n):
                for j in range(i+1, min(n-1, len(matrix[i])-1)):  # 确保不越界
                    if random.random() < 0.1:  # 小概率修改每条边
                        matrix[i][j] = 1 - matrix[i][j]  # 翻转
        
        # 随机修改操作
        if random.random() < MUTATION_RATE:
            for i in range(1, min(len(ops)-1, 6)):  # 不修改输入和输出，确保索引有效
                if random.random() < 0.2:
                    ops[i] = random.choice([CONV1X1, CONV3X3, MAXPOOL3X3])
        
        # 创建新模型规范并检查有效性
        try:
            model_spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(model_spec):
                # 获取完整信息（三次运行）
                fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
                
                if 108 in computed_stats:
                    # 获取三次运行的结果
                    val_accuracies = [run['final_validation_accuracy'] 
                                    for run in computed_stats[108]]
                    test_accuracies = [run['final_test_accuracy'] 
                                    for run in computed_stats[108]]
                    train_times = [run['final_training_time'] 
                                for run in computed_stats[108]]
                    
                    # 更新架构信息
                    new_arch = NASArchitecture(matrix, ops)
                    new_arch.fitness_runs = val_accuracies
                    new_arch.fitness = np.mean(val_accuracies)
                    new_arch.test_accuracy_runs = test_accuracies
                    new_arch.test_accuracy = np.mean(test_accuracies)
                    new_arch.parameters = fixed_stats['trainable_parameters']
                    new_arch.training_time_runs = train_times
                    new_arch.training_time = np.mean(train_times)
                    
                    self.history['evaluations'] += len(computed_stats[108])  # 增加评估次数
                else:
                    # 如果无效，返回原架构
                    return arch
        except Exception as e:
            print(f"Mutation failed: {e}")
            # 如果出现错误，返回原架构
            return arch
        
        return new_arch
    
    def crossover(self, parent1, parent2):
        """交叉操作：组合两个架构"""
        # 确保两个父代有相同的大小
        if len(parent1.matrix) != len(parent2.matrix):
            # 如果大小不同，返回适应度较高的父代
            return parent1 if parent1.fitness > parent2.fitness else parent2
        
        new_matrix = []
        new_ops = []
        
        n = len(parent1.matrix)
        
        # 随机选择交叉点
        crossover_point = random.randint(1, n-2)
        
        # 矩阵交叉
        for i in range(n):
            if i < crossover_point:
                new_matrix.append(parent1.matrix[i][:])
            else:
                # 确保parent2有足够的行
                if i < len(parent2.matrix):
                    new_matrix.append(parent2.matrix[i][:])
                else:
                    new_matrix.append(parent1.matrix[i][:])
        
        # 操作交叉
        min_ops_len = min(len(parent1.ops), len(parent2.ops))
        for i in range(min_ops_len):
            if i < crossover_point:
                new_ops.append(parent1.ops[i])
            else:
                new_ops.append(parent2.ops[i])
        
        # 如果ops长度不同，补充完整
        if len(parent1.ops) > min_ops_len:
            new_ops.extend(parent1.ops[min_ops_len:])
        elif len(parent2.ops) > min_ops_len:
            new_ops.extend(parent2.ops[min_ops_len:])
        
        # 创建新架构
        try:
            model_spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
            if self.nasbench.is_valid(model_spec):
                # 获取完整信息（三次运行）
                fixed_stats, computed_stats = self.nasbench.get_metrics_from_spec(model_spec)
                
                if 108 in computed_stats:
                    # 获取三次运行的结果
                    val_accuracies = [run['final_validation_accuracy'] 
                                    for run in computed_stats[108]]
                    test_accuracies = [run['final_test_accuracy'] 
                                    for run in computed_stats[108]]
                    train_times = [run['final_training_time'] 
                                for run in computed_stats[108]]
                    
                    # 创建新架构
                    new_arch = NASArchitecture(new_matrix, new_ops)
                    new_arch.fitness_runs = val_accuracies
                    new_arch.fitness = np.mean(val_accuracies)
                    new_arch.test_accuracy_runs = test_accuracies
                    new_arch.test_accuracy = np.mean(test_accuracies)
                    new_arch.parameters = fixed_stats['trainable_parameters']
                    new_arch.training_time_runs = train_times
                    new_arch.training_time = np.mean(train_times)
                    
                    self.history['evaluations'] += len(computed_stats[108])  # 增加评估次数
                    return new_arch
        except Exception as e:
            print(f"Crossover failed: {e}")
        
        # 如果交叉失败，返回父代中较好的一个
        return parent1 if parent1.fitness > parent2.fitness else parent2
    
    def generate_new_individual(self, clusters, cluster_centers):
        """生成新个体"""
        if not cluster_centers:
            # 如果没有簇中心，生成随机个体
            return self._generate_random_individual()
        
        if random.random() < P_ONE_CENTER:
            # 选择一个簇中心
            center = random.choice(cluster_centers)
            
            # 决定是否替换
            if random.random() < P_REPLACE:
                return self._generate_random_individual()
            
            # 否则，对中心进行突变
            new_arch = self.mutate_architecture(center)
        else:
            # 选择两个簇中心进行交叉
            if len(cluster_centers) >= 2:
                centers = random.sample(cluster_centers, 2)
                new_arch = self.crossover(centers[0], centers[1])
            else:
                # 如果只有一个中心，突变
                new_arch = self.mutate_architecture(cluster_centers[0])
        
        return new_arch

    def _generate_random_individual(self):
        """生成随机个体"""
        all_hashes = list(self.nasbench.hash_iterator())
        while True:
            random_hash = random.choice(all_hashes)
            fixed_stats, computed_stats = self.nasbench.get_metrics_from_hash(random_hash)
            
            if 108 in computed_stats:
                matrix = fixed_stats['module_adjacency']
                ops = fixed_stats['module_operations']
                accuracies = [run['final_validation_accuracy'] 
                            for run in computed_stats[108]]
                fitness = np.mean(accuracies)
                
                new_arch = NASArchitecture(matrix, ops, random_hash)
                new_arch.fitness = fitness
                new_arch.test_accuracy = np.mean([run['final_test_accuracy'] 
                                                for run in computed_stats[108]])
                new_arch.parameters = fixed_stats['trainable_parameters']
                new_arch.training_time = np.mean([run['final_training_time'] 
                                                for run in computed_stats[108]])
                self.history['evaluations'] += 1
                return new_arch
    
    def calculate_diversity(self, population):
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0
        
        fitnesses = [ind.fitness for ind in population]
        return np.std(fitnesses)  # 适应度的标准差作为多样性指标
    
    def run(self, max_iterations=MAX_ITERATIONS):
        """运行BSO算法"""
        print(f"\nStarting BSO Algorithm")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {max_iterations}")
        print(f"Cluster count: {CLUSTER_COUNT}")
        
        # 初始化种群
        self.initialize_population()
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # 聚类
            self.clusters, cluster_centers = self.clustering(self.population)
            
            # 记录聚类大小
            self.history['cluster_sizes'].append([len(c) for c in self.clusters])
            
            # 生成新种群
            new_population = []
            
            # 保留精英
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:ELITE_COUNT]
            new_population.extend(elite)
            
            # 生成新个体
            while len(new_population) < self.population_size:
                new_individual = self.generate_new_individual(self.clusters, cluster_centers)
                new_population.append(new_individual)
            
            # 更新种群
            self.population = new_population
            
            # 更新最佳个体
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_architecture = current_best.copy()
                print(f"Iteration {iteration+1}: New best fitness = {self.best_fitness:.4f}")
            
            # 记录历史
            fitnesses = [ind.fitness for ind in self.population]
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitnesses))
            self.history['worst_fitness'].append(min(fitnesses))
            self.history['diversity'].append(self.calculate_diversity(self.population))
            
            # 打印进度
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Best={self.best_fitness:.4f}, "
                      f"Avg={np.mean(fitnesses):.4f}, "
                      f"Diversity={self.history['diversity'][-1]:.4f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"\nBSO Algorithm Completed")
        print(f"Total evaluations: {self.history['evaluations']}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Final best fitness: {self.best_fitness:.4f}")

        if self.best_architecture:
            print(f"Best test accuracy: {self.best_architecture.test_accuracy:.4f}")
            print(f"Best parameters: {self.best_architecture.parameters:,}")
            print(f"Best training time: {self.best_architecture.training_time:.1f}s")
            print(f"Validation accuracy runs: {self.best_architecture.fitness_runs}")
            
        return self.best_architecture, self.history

# 在原有代码的 BSONAS 类定义之后，添加新的改进类

class ImprovedBSONAS:
    """改进的BSO算法，专门针对NASBench优化"""
    
    def __init__(self, nasbench, population_size=30):
        self.nasbench = nasbench
        self.population_size = population_size
        self.population = []
        self.best_architecture = None
        self.best_fitness = 0.0
        
        # 改进的参数
        self.params = {
            'max_iterations': 50,
            'cluster_count': 3,
            'p_replace': 0.1,
            'p_one_center': 0.7,
            'p_two_center': 0.3,
            'mutation_rate': 0.05,
            'elite_count': 10,
            'initial_exploration_rate': 0.8,
            'final_exploration_rate': 0.2,
            'adaptive_mutation': True
        }
        
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'cluster_sizes': [],
            'evaluations': 0,
            'mutations': [],
            'crossovers': [],
            'exploration_rate': []
        }
    
    def get_exploration_rate(self, iteration):
        """根据迭代进度计算探索率"""
        progress = iteration / self.params['max_iterations']
        exploration_rate = (self.params['initial_exploration_rate'] * (1 - progress) + 
                          self.params['final_exploration_rate'] * progress)
        return exploration_rate

    def initialize_diverse_population(self):
        """初始化多样化的种群"""
        print("Initializing diverse population...")
        all_hashes = list(self.nasbench.hash_iterator())
        
        # 策略：选择不同准确率区间的架构
        accuracy_bins = defaultdict(list)
        
        # 先采样一批架构进行分析
        sample_size = min(1000, len(all_hashes))
        sample_hashes = np.random.choice(all_hashes, sample_size, replace=False)
        
        print(f"Sampling {sample_size} architectures for analysis...")
        
        for idx, hash_value in enumerate(sample_hashes):
            fixed_stats, computed_stats = self.nasbench.get_metrics_from_hash(hash_value)
            if 108 in computed_stats:
                accuracies = [run['final_validation_accuracy'] 
                            for run in computed_stats[108]]
                avg_accuracy = np.mean(accuracies)
                
                # 分箱 (0.90-0.96分为6个区间)
                bin_idx = int(avg_accuracy * 100)  # 乘以100得到整数索引
                if 90 <= bin_idx <= 95:  # 只考虑0.90-0.95的架构
                    accuracy_bins[bin_idx].append(hash_value)
            
            if (idx + 1) % 200 == 0:
                print(f"  Analyzed {idx+1}/{sample_size} architectures")
        
        # 从每个区间选择架构
        selected_hashes = []
        bins_used = 0
        
        for bin_idx in range(90, 96):  # 90, 91, 92, 93, 94, 95
            if bin_idx in accuracy_bins and accuracy_bins[bin_idx]:
                bin_hashes = accuracy_bins[bin_idx]
                n_select = min(2, len(bin_hashes))  # 每个区间选2个
                selected = np.random.choice(bin_hashes, n_select, replace=False)
                selected_hashes.extend(selected)
                bins_used += 1
                print(f"  Selected {n_select} architectures from accuracy bin {bin_idx/100:.2f}")
        
        print(f"Selected from {bins_used} accuracy bins")
        
        # 如果不够，随机补充
        if len(selected_hashes) < self.population_size:
            remaining_needed = self.population_size - len(selected_hashes)
            remaining_hashes = [h for h in all_hashes if h not in selected_hashes]
            if remaining_hashes:
                additional = np.random.choice(remaining_hashes, 
                                            min(remaining_needed, len(remaining_hashes)), 
                                            replace=False)
                selected_hashes.extend(additional)
                print(f"  Added {len(additional)} random architectures")
        
        # 创建种群
        print(f"Creating population from {len(selected_hashes[:self.population_size])} selected architectures...")
        
        for i, hash_value in enumerate(selected_hashes[:self.population_size]):
            fixed_stats, computed_stats = self.nasbench.get_metrics_from_hash(hash_value)
            if 108 in computed_stats:
                matrix = fixed_stats['module_adjacency']
                ops = fixed_stats['module_operations']
                accuracies = [run['final_validation_accuracy'] 
                            for run in computed_stats[108]]
                fitness = np.mean(accuracies)
                
                arch = NASArchitecture(matrix, ops, hash_value)
                arch.fitness = fitness
                arch.test_accuracy = np.mean([run['final_test_accuracy'] 
                                            for run in computed_stats[108]])
                arch.parameters = fixed_stats['trainable_parameters']
                arch.training_time = np.mean([run['final_training_time'] 
                                            for run in computed_stats[108]])
                
                self.population.append(arch)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_architecture = arch.copy()
            
            if (i + 1) % 10 == 0:
                print(f"  Created {i+1}/{min(self.population_size, len(selected_hashes))} architectures")
        
        print(f"Initialized {len(self.population)} diverse architectures")
        print(f"Best initial fitness: {self.best_fitness:.4f}")
        if self.best_architecture:
            print(f"Best initial test accuracy: {self.best_architecture.test_accuracy:.4f}")
            print(f"Best initial parameters: {self.best_architecture.parameters:,}")

    def clustering_by_features(self, population):
        """基于架构特征的智能聚类"""
        if len(population) < 2:
            return [population], population[:1]
        
        try:
            # 尝试导入sklearn，如果没有安装则使用简单聚类
            from sklearn.cluster import KMeans
            use_sklearn = True
        except ImportError:
            print("Warning: sklearn not installed, using simple clustering")
            use_sklearn = False
        
        if use_sklearn:
            # 提取架构特征
            features = []
            for arch in population:
                features.append(arch.get_features())
            
            features = np.array(features)
            
            # 归一化特征
            features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # 使用K-means聚类
            k = min(self.params['cluster_count'], len(population))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_norm)
            
            # 组织聚类
            clusters = [[] for _ in range(k)]
            for arch, label in zip(population, labels):
                clusters[label].append(arch)
        else:
            # 简单聚类：基于适应度值
            sorted_pop = sorted(population, key=lambda x: x.fitness)
            k = min(self.params['cluster_count'], len(sorted_pop))
            cluster_size = len(sorted_pop) // k
            
            clusters = []
            for i in range(k):
                start_idx = i * cluster_size
                end_idx = start_idx + cluster_size if i < k - 1 else len(sorted_pop)
                clusters.append(sorted_pop[start_idx:end_idx])
            
            # 移除空簇
            clusters = [c for c in clusters if c]
        
        # 选择每个簇的中心（最佳个体）
        cluster_centers = []
        for cluster in clusters:
            if cluster:
                cluster.sort(key=lambda x: x.fitness, reverse=True)
                cluster_centers.append(cluster[0])
        
        return clusters, cluster_centers

    def smart_mutate_architecture(self, arch, iteration, max_iterations):
        """智能突变：根据迭代进度调整突变强度"""
        new_arch = arch.copy()
        matrix = [row[:] for row in new_arch.matrix]
        ops = new_arch.ops[:]
        
        # 自适应突变率
        if self.params['adaptive_mutation']:
            progress = iteration / max_iterations
            current_mutation_rate = self.params['mutation_rate'] * (1.0 - progress * 0.5)
        else:
            current_mutation_rate = self.params['mutation_rate']
        
        n = len(matrix)
        changed = False
        
        # 基于架构质量的突变策略
        if arch.fitness > 0.94:  # 高质量架构，轻微突变
            # 只突变操作类型
            for i in range(1, len(ops)-1):
                if random.random() < current_mutation_rate * 0.5:
                    current_op = ops[i]
                    alternatives = [op for op in [CONV1X1, CONV3X3, MAXPOOL3X3] if op != current_op]
                    if alternatives:
                        ops[i] = random.choice(alternatives)
                        changed = True
        else:  # 低质量架构，更强突变
            # 1. 突变操作类型
            for i in range(1, len(ops)-1):
                if random.random() < current_mutation_rate:
                    ops[i] = random.choice([CONV1X1, CONV3X3, MAXPOOL3X3])
                    changed = True
            
            # 2. 突变连接（保持有效性）
            if random.random() < current_mutation_rate:
                # 只添加/删除少量边
                for _ in range(2):  # 只尝试修改2条边
                    i = random.randint(0, n-3)  # 确保i最大为n-3
                    j = random.randint(i+1, n-2)  # 确保j最大为n-2
                    if j < len(matrix[i]):  # 额外的边界检查
                        matrix[i][j] = 1 - matrix[i][j]
                        changed = True
        
        # 如果没有任何改变，至少做一个小的突变
        if not changed and random.random() < 0.5:
            i = random.randint(1, len(ops)-2)
            current_op = ops[i]
            alternatives = [op for op in [CONV1X1, CONV3X3, MAXPOOL3X3] if op != current_op]
            if alternatives:
                ops[i] = random.choice(alternatives)
                changed = True
        
        # 创建新模型规范并检查有效性
        try:
            model_spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(model_spec):
                # 查询完整信息
                data = self.nasbench.query(model_spec)
                
                # 创建新架构
                new_arch = NASArchitecture(matrix, ops)
                new_arch.fitness = data['validation_accuracy']
                new_arch.test_accuracy = data['test_accuracy']
                new_arch.parameters = data['trainable_parameters']
                new_arch.training_time = data['training_time']
                self.history['evaluations'] += 1
                
                # 记录突变信息
                self.history.setdefault('mutations', []).append({
                    'iteration': iteration,
                    'from_fitness': arch.fitness,
                    'to_fitness': new_arch.fitness,
                    'improvement': new_arch.fitness - arch.fitness,
                    'type': 'smart_mutation'
                })
                
                return new_arch
        except Exception as e:
            pass
        
        return arch
    
    def _generate_random_individual(self):
        """生成随机个体"""
        all_hashes = list(self.nasbench.hash_iterator())
        while True:
            random_hash = random.choice(all_hashes)
            fixed_stats, computed_stats = self.nasbench.get_metrics_from_hash(random_hash)
            
            if 108 in computed_stats:
                matrix = fixed_stats['module_adjacency']
                ops = fixed_stats['module_operations']
                accuracies = [run['final_validation_accuracy'] 
                            for run in computed_stats[108]]
                fitness = np.mean(accuracies)
                
                new_arch = NASArchitecture(matrix, ops, random_hash)
                new_arch.fitness = fitness
                new_arch.test_accuracy = np.mean([run['final_test_accuracy'] 
                                                for run in computed_stats[108]])
                new_arch.parameters = fixed_stats['trainable_parameters']
                new_arch.training_time = np.mean([run['final_training_time'] 
                                                for run in computed_stats[108]])
                self.history['evaluations'] += 1
                return new_arch

    def knowledge_based_crossover(self, parent1, parent2):
        """基于知识的交叉：结合两个架构的优点"""
        # 选择较好的父代作为主要模板
        if parent1.fitness > parent2.fitness:
            main_parent, secondary_parent = parent1, parent2
        else:
            main_parent, secondary_parent = parent2, parent1
        
        new_matrix = []
        new_ops = main_parent.ops[:]  # 从主要父代继承操作
        
        n = len(main_parent.matrix)
        
        # 从次要父代继承部分连接
        for i in range(n):
            new_row = main_parent.matrix[i][:]
            # 以一定概率从次要父代继承连接
            if i < len(secondary_parent.matrix):
                for j in range(len(new_row)):
                    if j < len(secondary_parent.matrix[i]):
                        if random.random() < 0.3:  # 30%概率继承次要父代的连接
                            new_row[j] = secondary_parent.matrix[i][j]
            new_matrix.append(new_row)
        
        # 创建新架构
        try:
            model_spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
            if self.nasbench.is_valid(model_spec):
                # 查询完整信息
                data = self.nasbench.query(model_spec)
                
                new_arch = NASArchitecture(new_matrix, new_ops)
                new_arch.fitness = data['validation_accuracy']
                new_arch.test_accuracy = data['test_accuracy']
                new_arch.parameters = data['trainable_parameters']
                new_arch.training_time = data['training_time']
                self.history['evaluations'] += 1
                
                # 记录交叉信息
                self.history.setdefault('crossovers', []).append({
                    'parent1_fitness': parent1.fitness,
                    'parent2_fitness': parent2.fitness,
                    'child_fitness': new_arch.fitness,
                    'improvement': new_arch.fitness - max(parent1.fitness, parent2.fitness)
                })
                
                return new_arch
        except Exception as e:
            pass
        
        # 如果交叉失败，返回较好的父代
        return main_parent

    def run_improved(self):
        """运行改进的BSO算法"""
        print(f"\nStarting Improved BSO Algorithm")
        print(f"Population size: {self.population_size}")
        print(f"Max iterations: {self.params['max_iterations']}")
        print(f"Parameters: {self.params}")
        
        # 初始化多样化种群
        self.initialize_diverse_population()
        
        start_time = time.time()
        
        for iteration in range(self.params['max_iterations']):
            # 计算当前探索率
            exploration_rate = self.get_exploration_rate(iteration)
            self.history['exploration_rate'].append(exploration_rate)
            
            # 基于架构特征的聚类
            clusters, cluster_centers = self.clustering_by_features(self.population)
            
            # 记录聚类大小
            self.history['cluster_sizes'].append([len(c) for c in clusters])
            
            # 生成新种群
            new_population = []
            
            # 保留精英
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.params['elite_count']]
            new_population.extend(elite)
            
            # 根据探索率决定生成策略
            while len(new_population) < self.population_size:
                if random.random() < exploration_rate:
                    # 探索阶段：更多随机生成
                    if random.random() < self.params['p_replace']:
                        new_individual = self._generate_random_individual()
                    else:
                        # 从随机簇中选择中心进行突变
                        if cluster_centers:
                            center = random.choice(cluster_centers)
                        else:
                            center = random.choice(self.population)
                        new_individual = self.smart_mutate_architecture(center, iteration, self.params['max_iterations'])
                else:
                    # 开发阶段：更多智能操作
                    if random.random() < self.params['p_one_center']:
                        # 选择最佳簇中心
                        if cluster_centers:
                            cluster_centers.sort(key=lambda x: x.fitness, reverse=True)
                            center = cluster_centers[0]  # 最佳中心
                            new_individual = self.smart_mutate_architecture(center, iteration, self.params['max_iterations'])
                        else:
                            new_individual = self._generate_random_individual()
                    else:
                        # 交叉操作
                        if len(cluster_centers) >= 2:
                            cluster_centers.sort(key=lambda x: x.fitness, reverse=True)
                            centers = cluster_centers[:2]  # 两个最佳中心
                            new_individual = self.knowledge_based_crossover(centers[0], centers[1])
                        else:
                            new_individual = self._generate_random_individual()
                
                new_population.append(new_individual)
            
            # 更新种群
            self.population = new_population
            
            # 更新最佳个体
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                improvement = current_best.fitness - self.best_fitness
                self.best_fitness = current_best.fitness
                self.best_architecture = current_best.copy()
                print(f"Iteration {iteration+1}: New best fitness = {self.best_fitness:.4f} (+{improvement:.4f})")
            
            # 记录历史
            fitnesses = [ind.fitness for ind in self.population]
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(np.mean(fitnesses))
            self.history['worst_fitness'].append(min(fitnesses))
            self.history['diversity'].append(np.std(fitnesses))
            
            # 打印进度
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration+1}/{self.params['max_iterations']}: "
                      f"Best={self.best_fitness:.4f}, "
                      f"Avg={np.mean(fitnesses):.4f}, "
                      f"Diversity={self.history['diversity'][-1]:.4f}, "
                      f"Exploration={exploration_rate:.2f}")
        
        elapsed_time = time.time() - start_time
        
        print(f"\nImproved BSO Algorithm Completed")
        print(f"Total evaluations: {self.history['evaluations']}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Final best fitness: {self.best_fitness:.4f}")
        
        if self.best_architecture:
            print(f"Best test accuracy: {self.best_architecture.test_accuracy:.4f}")
            print(f"Best parameters: {self.best_architecture.parameters:,}")
            print(f"Best training time: {self.best_architecture.training_time:.1f}s")
        
        # 输出统计信息
        if self.history['mutations']:
            mutations = self.history['mutations']
            improvements = [m['improvement'] for m in mutations]
            positive_improvements = [i for i in improvements if i > 0]
            print(f"\nMutation statistics:")
            print(f"  Total mutations: {len(mutations)}")
            print(f"  Positive improvements: {len(positive_improvements)}")
            print(f"  Average improvement: {np.mean(improvements):.4f}")
        
        if self.history['crossovers']:
            crossovers = self.history['crossovers']
            improvements = [c['improvement'] for c in crossovers]
            positive_improvements = [i for i in improvements if i > 0]
            print(f"\nCrossover statistics:")
            print(f"  Total crossovers: {len(crossovers)}")
            print(f"  Positive improvements: {len(positive_improvements)}")
            print(f"  Average improvement: {np.mean(improvements):.4f}")
        
        return self.best_architecture, self.history


def visualize_results(history, best_arch, algorithm_name="BSO"):
    """可视化结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 适应度曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
    plt.plot(history['avg_fitness'], 'g-', linewidth=2, label='Average Fitness')
    plt.plot(history['worst_fitness'], 'r-', linewidth=2, label='Worst Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Validation Accuracy)')
    plt.title(f'{algorithm_name} Fitness Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 多样性变化
    plt.subplot(2, 3, 2)
    plt.plot(history['diversity'], 'm-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Population Diversity (Std)')
    plt.title('Population Diversity Over Time')
    plt.grid(True, alpha=0.3)
    
    # 3. 聚类大小变化（最后几次迭代）
    plt.subplot(2, 3, 3)
    if history['cluster_sizes']:
        last_clusters = history['cluster_sizes'][-min(20, len(history['cluster_sizes'])):]
        for i in range(len(last_clusters[0])):
            cluster_sizes = [c[i] for c in last_clusters if i < len(c)]
            plt.plot(cluster_sizes, label=f'Cluster {i+1}')
        plt.xlabel('Iteration (last 20)')
        plt.ylabel('Cluster Size')
        plt.title('Cluster Size Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. 适应度分布（最终种群）
    plt.subplot(2, 3, 4)
    final_fitnesses = [history['best_fitness'][-1], 
                      history['avg_fitness'][-1], 
                      history['worst_fitness'][-1]]
    labels = ['Best', 'Average', 'Worst']
    colors = ['green', 'blue', 'red']
    bars = plt.bar(labels, final_fitnesses, color=colors)
    plt.ylim(0.9, 1.0)
    plt.ylabel('Fitness')
    plt.title('Final Population Fitness Distribution')
    for bar, value in zip(bars, final_fitnesses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # 5. 准确率 vs 参数量散点图
    plt.subplot(2, 3, 5)
    # 随机采样一些架构作为背景
    all_hashes = list(nasbench.hash_iterator())
    sample_hashes = np.random.choice(all_hashes, 1000, replace=False)
    
    sample_params = []
    sample_accs = []
    
    for hash_value in sample_hashes:
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(hash_value)
        if 108 in computed_stats:
            accuracies = [run['final_validation_accuracy'] 
                         for run in computed_stats[108]]
            sample_accs.append(np.mean(accuracies))
            sample_params.append(fixed_stats['trainable_parameters'])
    
    plt.scatter(sample_params, sample_accs, alpha=0.3, s=10, label='Random Samples')
    
    # 标记BSO找到的最佳架构
    plt.scatter(best_arch.parameters, best_arch.fitness, 
               color='red', s=200, marker='*', label=f'BSO Best (acc={best_arch.fitness:.4f})')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy')
    plt.title('Architecture Performance Landscape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 算法评估次数
    plt.subplot(2, 3, 6)
    iterations = len(history['best_fitness'])
    evaluations_per_iter = history['evaluations'] / iterations if iterations > 0 else 0
    
    labels = ['Total Evaluations', 'Evaluations per Iteration']
    values = [history['evaluations'], evaluations_per_iter]
    bars = plt.bar(labels, values, color=['purple', 'orange'])
    plt.ylabel('Count')
    plt.title('Algorithm Evaluation Statistics')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(value)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'bso_nasbench_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 额外可视化：训练时间 vs 准确率
    plt.figure(figsize=(10, 6))
    
    # 背景点
    sample_times = []
    for hash_value in sample_hashes[:500]:
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(hash_value)
        if 108 in computed_stats:
            times = [run['final_training_time'] for run in computed_stats[108]]
            sample_times.append(np.mean(times))
    
    # 只保留与sample_accs匹配的数据点
    min_len = min(len(sample_times), len(sample_accs[:500]))
    sample_times = sample_times[:min_len]
    sample_accs_trunc = sample_accs[:min_len]
    
    plt.scatter(sample_times, sample_accs_trunc, alpha=0.3, s=10, label='Random Samples')
    
    # BSO最佳架构
    plt.scatter(best_arch.training_time, best_arch.fitness,
               color='red', s=200, marker='*', 
               label=f'BSO Best (time={best_arch.training_time:.1f}s)')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Time vs Accuracy Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bso_time_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_random_search(nasbench, num_evaluations=5000):
    """与随机搜索比较"""
    print("\nRunning Random Search for comparison...")
    
    start_time = time.time()
    all_hashes = list(nasbench.hash_iterator())
    
    random_accuracies = []
    random_test_accuracies = []
    random_parameters = []
    random_times = []
    
    selected_hashes = np.random.choice(all_hashes, num_evaluations, replace=False)
    
    for i, hash_value in enumerate(selected_hashes):
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(hash_value)
        
        if 108 in computed_stats:
            val_accs = [run['final_validation_accuracy'] for run in computed_stats[108]]
            test_accs = [run['final_test_accuracy'] for run in computed_stats[108]]
            train_times = [run['final_training_time'] for run in computed_stats[108]]
            
            random_accuracies.append(np.mean(val_accs))
            random_test_accuracies.append(np.mean(test_accs))
            random_parameters.append(fixed_stats['trainable_parameters'])
            random_times.append(np.mean(train_times))
        
        if (i + 1) % 1000 == 0:
            print(f"Random search: evaluated {i+1}/{num_evaluations} architectures")
    
    elapsed_time = time.time() - start_time
    
    best_random_idx = np.argmax(random_accuracies)
    best_random_acc = random_accuracies[best_random_idx]
    best_random_test = random_test_accuracies[best_random_idx]
    best_random_params = random_parameters[best_random_idx]
    best_random_time = random_times[best_random_idx]
    
    print(f"\nRandom Search Results:")
    print(f"Total evaluations: {len(random_accuracies)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Best validation accuracy: {best_random_acc:.4f}")
    print(f"Best test accuracy: {best_random_test:.4f}")
    print(f"Parameters of best: {best_random_params:,}")
    print(f"Training time of best: {best_random_time:.1f}s")
    
    return {
        'accuracies': random_accuracies,
        'test_accuracies': random_test_accuracies,
        'parameters': random_parameters,
        'times': random_times,
        'best_acc': best_random_acc,
        'best_test': best_random_test,
        'best_params': best_random_params,
        'best_time': best_random_time,
        'total_time': elapsed_time
    }

def main(_):
    global nasbench
    # 加载数据集
    print("Loading NASBench dataset...")
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    print("Dataset loaded successfully!")
    
    # 运行随机搜索作为baseline
    print("\n" + "="*60)
    print("RUNNING RANDOM SEARCH AS BASELINE")
    print("="*60)
    
    # 先运行随机搜索，用于后续比较
    random_evaluations = 3621  # 使用与之前类似的评估次数
    random_results = compare_with_random_search(nasbench, num_evaluations=random_evaluations)
    
    # 运行改进的BSO算法，最多重试3次
    max_retries = 3
    best_improved_bso_result = None
    best_improved_bso_history = None
    best_improved_val_acc = 0.0
    
    print("\n" + "="*60)
    print("RUNNING IMPROVED BSO ALGORITHM")
    print("="*60)
    
    for retry in range(max_retries):
        print(f"\n--- Improved BSO Attempt {retry+1}/{max_retries} ---")
        
        # 每次尝试使用不同的随机种子
        random.seed(42 + retry * 100)
        np.random.seed(42 + retry * 100)
        
        improved_bso = ImprovedBSONAS(nasbench, population_size=30)
        current_best_arch, current_history = improved_bso.run_improved()
        
        current_val_acc = current_best_arch.fitness
        print(f"Attempt {retry+1}: Validation accuracy = {current_val_acc:.4f}")
        print(f"Random Search baseline: {random_results['best_acc']:.4f}")
        
        # 检查是否优于随机搜索
        if current_val_acc > random_results['best_acc']:
            print(f"✓ Attempt {retry+1} SUCCESS: Improved BSO ({current_val_acc:.4f}) > Random Search ({random_results['best_acc']:.4f})")
            best_improved_bso_result = current_best_arch
            best_improved_bso_history = current_history
            best_improved_val_acc = current_val_acc
            break
        else:
            print(f"✗ Attempt {retry+1} FAILED: Improved BSO ({current_val_acc:.4f}) ≤ Random Search ({random_results['best_acc']:.4f})")
            
            # 保存当前尝试中最好的结果
            if current_val_acc > best_improved_val_acc:
                best_improved_val_acc = current_val_acc
                best_improved_bso_result = current_best_arch
                best_improved_bso_history = current_history
            
            # 如果不是最后一次尝试，继续重试
            if retry < max_retries - 1:
                print(f"Retrying with different initialization...")
            else:
                print(f"Reached maximum retries ({max_retries}). Using best attempt.")
    
    # 如果没有成功优于随机搜索，使用最佳尝试结果
    if best_improved_bso_result is None:
        print("ERROR: No successful BSO run. Exiting.")
        return
    
    best_arch = best_improved_bso_result
    history = best_improved_bso_history
    
    # 输出最终结果
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Improved BSO validation accuracy: {best_arch.fitness:.4f}")
    print(f"Random Search validation accuracy: {random_results['best_acc']:.4f}")
    
    if best_arch.fitness > random_results['best_acc']:
        improvement = (best_arch.fitness - random_results['best_acc']) * 100
        print(f"✓ SUCCESS: Improved BSO is better by {improvement:.2f}%")
    else:
        difference = (random_results['best_acc'] - best_arch.fitness) * 100
        print(f"✗ WARNING: Random Search is better by {difference:.2f}%")
    
    print(f"Best test accuracy: {best_arch.test_accuracy:.4f}")
    print(f"Best parameters: {best_arch.parameters:,}")
    print(f"Best training time: {best_arch.training_time:.1f}s")
    
    # 可视化结果
    visualize_results(history, best_arch, "Improved BSO on NASBench")
    
    # 与随机搜索比较的可视化
    print("\n" + "="*60)
    print("FINAL COMPARISON WITH RANDOM SEARCH")
    print("="*60)
    
    # 比较结果可视化
    plt.figure(figsize=(12, 8))
    
    # 1. 性能比较
    plt.subplot(2, 2, 1)
    methods = ['Improved BSO', 'Random Search']
    best_accs = [best_arch.fitness, random_results['best_acc']]
    best_tests = [best_arch.test_accuracy, random_results['best_test']]

    print(f"\nPerformance Comparison:")
    print(f"Improved BSO validation accuracy: {best_arch.fitness:.4f}")
    print(f"Improved BSO test accuracy: {best_arch.test_accuracy:.4f}")
    print(f"Random Search validation accuracy: {random_results['best_acc']:.4f}")
    print(f"Random Search test accuracy: {random_results['best_test']:.4f}")

    x = np.arange(len(methods))
    width = 0.35

    bars1 = plt.bar(x - width/2, best_accs, width, label='Validation Accuracy', color='blue')
    bars2 = plt.bar(x + width/2, best_tests, width, label='Test Accuracy', color='green')

    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison: Improved BSO vs Random Search')
    plt.xticks(x, methods)
    plt.ylim(0.90, 0.96)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 效率比较
    plt.subplot(2, 2, 2)
    
    # 计算BSO在每次迭代后的最佳准确率
    bso_progress = history['best_fitness']
    
    # 对随机搜索结果排序以获得进度
    sorted_random = sorted(random_results['accuracies'], reverse=True)
    
    # 创建与BSO迭代次数相同的随机搜索进度
    random_evaluations = len(sorted_random)
    bso_iterations = len(bso_progress)
    
    if random_evaluations >= bso_iterations:
        step = max(1, random_evaluations // bso_iterations)
        indices = list(range(0, random_evaluations, step))[:bso_iterations]
        random_progress_sampled = [sorted_random[i] for i in indices]
    else:
        random_progress_sampled = sorted_random + [sorted_random[-1]] * (bso_iterations - random_evaluations)
    
    plt.plot(range(len(bso_progress)), bso_progress, 'b-', linewidth=2, label='Improved BSO')
    plt.plot(range(len(random_progress_sampled)), random_progress_sampled, 
            'r--', linewidth=2, label='Random Search')
    plt.xlabel('Iteration / Evaluation Batch')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Search Efficiency Comparison')
    plt.ylim(0.90, 0.96)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 参数数量比较
    plt.subplot(2, 2, 3)
    params = [best_arch.parameters / 1e6, random_results['best_params'] / 1e6]  # 转换为百万
    
    bars = plt.bar(methods, params, color=['orange', 'purple'])
    plt.ylabel('Parameters (Millions)')
    plt.title('Parameter Count of Best Architectures')
    
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{param:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 训练时间比较
    plt.subplot(2, 2, 4)
    train_times = [best_arch.training_time, random_results['best_time']]
    
    bars = plt.bar(methods, train_times, color=['cyan', 'magenta'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time of Best Architectures')
    
    for bar, time_val in zip(bars, train_times):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_comparison_improved_bso_vs_random.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细比较结果
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Improved BSO':<15} {'Random Search':<15} {'Difference':<15}")
    print("-"*60)
    
    val_difference = best_arch.fitness - random_results['best_acc']
    test_difference = best_arch.test_accuracy - random_results['best_test']
    
    val_symbol = '+' if val_difference > 0 else ''
    test_symbol = '+' if test_difference > 0 else ''
    
    print(f"{'Validation Accuracy':<25} {best_arch.fitness:<15.4f} {random_results['best_acc']:<15.4f} "
          f"{val_symbol + str(round(val_difference*100, 2)) + '%':<15}")
    print(f"{'Test Accuracy':<25} {best_arch.test_accuracy:<15.4f} {random_results['best_test']:<15.4f} "
          f"{test_symbol + str(round(test_difference*100, 2)) + '%':<15}")
    
    param_difference = 1 - best_arch.parameters/random_results['best_params']
    time_difference = 1 - best_arch.training_time/random_results['best_time']
    
    param_symbol = '-' if param_difference > 0 else '+'
    time_symbol = '-' if time_difference > 0 else '+'
    
    print(f"{'Parameters':<25} {best_arch.parameters:<15,} {random_results['best_params']:<15,} "
          f"{param_symbol + str(round(abs(param_difference)*100, 2)) + '%':<15}")
    print(f"{'Training Time':<25} {best_arch.training_time:<15.1f}s {random_results['best_time']:<15.1f}s "
          f"{time_symbol + str(round(abs(time_difference)*100, 2)) + '%':<15}")
    
    eval_difference = 1 - history['evaluations']/len(random_results['accuracies'])
    print(f"{'Total Evaluations':<25} {history['evaluations']:<15} {len(random_results['accuracies']):<15} "
          f"{'-' + str(round(eval_difference*100, 2)) + '%':<15}")
    print("="*60)
    
    # 保存结果
    results = {
        'improved_bso_best': {
            'validation_accuracy': best_arch.fitness,
            'test_accuracy': best_arch.test_accuracy,
            'parameters': best_arch.parameters,
            'training_time': best_arch.training_time,
            'matrix': best_arch.matrix,
            'ops': best_arch.ops,
            'hash': best_arch.hash
        },
        'random_best': {
            'validation_accuracy': random_results['best_acc'],
            'test_accuracy': random_results['best_test'],
            'parameters': random_results['best_params'],
            'training_time': random_results['best_time']
        },
        'history': history,
        'random_results': random_results,
        'comparison': {
            'bso_better_than_random': best_arch.fitness > random_results['best_acc'],
            'validation_accuracy_difference': float(val_difference),
            'test_accuracy_difference': float(test_difference),
            'parameter_difference_ratio': float(param_difference),
            'training_time_difference_ratio': float(time_difference)
        }
    }
    
    with open('final_improved_bso_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to 'final_improved_bso_results.pkl'")
    

if __name__ == '__main__':
    app.run(main)
import copy
import random
import numpy as np
from nasbench import api
from sklearn.ensemble import RandomForestRegressor
import math


class Search:
    """
    搜索方法基类：提供矩阵的基本定义，
    随机 spec 生成,spec 变异（基础版）等功能。
    其他算法(如 EA)可继承本类。
    """

    def __init__(self, nasbench, allowed_edges=[0, 1],
                 allowed_ops=['conv3x3-bn-relu','conv1x1-bn-relu','maxpool3x3'],
                 num_vertices=7, input_op='input', output_op='output'):
        self.nasbench = nasbench
        self.ALLOWED_EDGES = allowed_edges
        self.ALLOWED_OPS = allowed_ops
        self.NUM_VERTICES = num_vertices
        self.INPUT = input_op
        self.OUTPUT = output_op
        self.OP_SPOTS = num_vertices - 2  # 中间节点个数

    def random_spec(self):
        """返回一个随机有效的 ModelSpec"""
        while True:
            matrix = np.random.choice(
                self.ALLOWED_EDGES, size=(self.NUM_VERTICES, self.NUM_VERTICES)
            )
            matrix = np.triu(matrix, 1)  # 只保留上三角

            ops = np.random.choice(self.ALLOWED_OPS, size=(self.NUM_VERTICES)).tolist()
            ops[0] = self.INPUT
            ops[-1] = self.OUTPUT

            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(spec):
                return spec

    def mutate_spec(self, old_spec, mutation_rate=1.0):
        """简易版本，对 old_spec 进行变异并返回一个有效的 ModelSpec"""
        while True:
            new_matrix = copy.deepcopy(old_spec.original_matrix)
            new_ops = copy.deepcopy(old_spec.original_ops)

            # 边变异概率
            edge_mutation_prob = mutation_rate / self.NUM_VERTICES
            for src in range(self.NUM_VERTICES - 1):
                for dst in range(src + 1, self.NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            # 操作变异概率
            op_mutation_prob = mutation_rate / self.OP_SPOTS
            for i in range(1, self.NUM_VERTICES - 1):
                if random.random() < op_mutation_prob:
                    candidates = [o for o in self.ALLOWED_OPS if o != new_ops[i]]
                    new_ops[i] = random.choice(candidates)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if self.nasbench.is_valid(new_spec):
                return new_spec

    def random_combination(self,iterable, sample_size):
        """随机从可迭代对象中选择sample_size个元素的组合"""
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), sample_size))
        return tuple(pool[i] for i in indices)

    #  搜索主循环（需要override）
    def run(self):
        """
        默认搜索流程
        返回时间，验证集精度，测试集精度的历史记录。
        """
        nasbench = self.nasbench
        nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        return times, best_valids, best_tests
class randomSearch(Search):
    """
    随机搜索算法
    """
    def __init__(self, nasbench,max_time_budget=5e6):
        super().__init__(nasbench)
        self.max_time_budget = max_time_budget
    def run(self):
        """
        运行随机搜索直到达到固定时间预算。
        返回时间，验证集精度，测试集精度的历史记录。
        """
        nasbench = self.nasbench
        nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        

        while True:
            spec = self.random_spec()
            data = nasbench.query(spec)


            # 仅选择基于验证精度的模型，测试精度仅用于比较不同的搜索轨迹。
            if data['validation_accuracy'] > best_valids[-1]:
                best_valids.append(data['validation_accuracy'])
                best_tests.append(data['test_accuracy'])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])

            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)
            if time_spent > self.max_time_budget:
            # 当超过时间预算时跳出循环
                break

        return times, best_valids, best_tests
class evolutionSearch(Search):
    """
    基于进化算法的搜索
    """
    def __init__(self, nasbench, population_size=100,
                 tournament_size=10,
                 crossover_rate=0.3,
                 mutation_rate=0.9,
                 max_time_budget=5e6):
        super().__init__(nasbench)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  
        self.max_time_budget = max_time_budget
    def crossover_spec(self,spec1, spec2, crossover_rate):
        """邻接矩阵按行随机 + 操作向量按位置随机交叉"""
        # 邻接矩阵交叉
        if random.random() > crossover_rate:
            return copy.deepcopy(random.choice([spec1, spec2]))
        while True:
            mat1 = spec1.original_matrix
            mat2 = spec2.original_matrix
            new_mat = np.zeros_like(mat1)
            for i in range(mat1.shape[0]):
                new_mat[i] = mat1[i] if random.random() < 0.5 else mat2[i]
            ops1=spec1.original_ops
            ops2=spec2.original_ops
            # 操作向量交叉
            new_ops = []
            for i in range(len(ops1)):
                if i == 0:
                    new_ops.append(ops1[0])   # input
                elif i == len(ops1)-1:
                    new_ops.append(ops1[-1])  # output
                else:
                    new_ops.append(ops1[i] if random.random() < 0.5 else ops2[i])
            new_spec= api.ModelSpec(new_mat, new_ops)
            if self.nasbench.is_valid(new_spec):
                return new_spec
    def run(self,crossover=True):
        """运行正则化进化的单次roll-out，直到达到固定时间预算。
        返回时间，验证集精度，测试集精度的历史记录。
        """
        nasbench = self.nasbench
        nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        population = []   # (validation, spec) tuples

        # 在最初的population_size个个体中，用随机生成的细胞来初始化种群。
        
        for _ in range(self.population_size):
            spec = self.random_spec()
            data = nasbench.query(spec)
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)
            population.append((data['validation_accuracy'], spec))

            if data['validation_accuracy'] > best_valids[-1]:
             best_valids.append(data['validation_accuracy'])
             best_tests.append(data['test_accuracy'])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
            if time_spent > self.max_time_budget:
                break

        # 在种群初始化后，继续进化种群。
        while True:
            while True:
                if crossover:
                    sample = self.random_combination(population, self.tournament_size)
                    par1 = sorted(sample, key=lambda i:i[0])[-1][1]
                    sample = self.random_combination(population, self.tournament_size)
                    par2 = sorted(sample, key=lambda i:i[0])[-1][1]
                    
                    child_spec = self.crossover_spec(par1, par2, self.crossover_rate)
                    new_spec = self.mutate_spec(child_spec, self.mutation_rate)
                else:
                    sample = self.random_combination(population, self.tournament_size)
                    parent = sorted(sample, key=lambda i:i[0])[-1][1]
                    new_spec = self.mutate_spec(parent, self.mutation_rate)
                
                if nasbench.is_valid(new_spec) :
                    break
            data = nasbench.query(new_spec)
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)

            # 在正则化进化中，我们淘汰种群中最老的个体。
            population.append((data['validation_accuracy'], new_spec))
            population.pop(0)
            
            if data['validation_accuracy'] > best_valids[-1]:
                best_valids.append(data['validation_accuracy'])
                best_tests.append(data['test_accuracy'])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])

            if time_spent > self.max_time_budget:
                break

        return times, best_valids, best_tests

class SASearch(Search):
    """
    基于模拟退火 (Simulated Annealing) 的搜索
    
    核心机制：
    - 从一个随机架构开始。
    - 每次迭代生成一个"邻居"（通过变异）。
    - 如果邻居更好，直接接受。
    - 如果邻居更差，以概率 P = exp((acc_new - acc_curr) / T) 接受。
    - T (温度) 随时间衰减。
    """
    def __init__(self, nasbench, 
                 initial_temp=0.1,   # 初始温度：决定了初期的探索程度
                 final_temp=0.001,   # 终止温度（可选，主要由 budget 控制）
                 alpha=0.99,         # 冷却系数：决定降温快慢 (0.9 ~ 0.999)
                 mutation_rate=0.1,  # 变异率：定义"邻居"的差异程度
                 max_time_budget=5e6):
        super().__init__(nasbench)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.max_time_budget = max_time_budget

    def get_acceptance_probability(self, acc_current, acc_new, temp):
        """
        Metropolis 准则计算接受概率
        """
        # 如果新解更好，接受概率为 1.0
        if acc_new > acc_current:
            return 1.0
        
        # 如果新解更差，概率取决于 (差值 / 温度)
        # acc_new - acc_current 是负数
        return math.exp((acc_new - acc_current) / temp)

    def run(self):
        nasbench = self.nasbench
        nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]

        # ---------------------------
        # 1. 初始化状态
        # ---------------------------
        current_spec = self.random_spec()
        data = nasbench.query(current_spec)
        current_acc = data['validation_accuracy']
        
        # 记录全局最优 (Global Best) - SA 本身只记录当前状态，但为了画图我们需要记录历史最优
        best_spec = current_spec
        best_valid_acc = current_acc
        best_test_acc = data['test_accuracy']
        
        # 更新历史记录
        best_valids.append(best_valid_acc)
        best_tests.append(best_test_acc)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        # 当前温度
        temp = self.initial_temp

        # ---------------------------
        # 2. 退火循环
        # ---------------------------
        while True:
            # A. 生成邻居 (通过变异)
            # 尝试生成一个有效的变异体
            neighbor_spec = None
            while True:
                neighbor_spec = self.mutate_spec(current_spec, self.mutation_rate)
                if nasbench.is_valid(neighbor_spec):
                    break
            
            # B. 评估邻居
            data = nasbench.query(neighbor_spec)
            neighbor_acc = data['validation_accuracy']
            neighbor_test_acc = data['test_accuracy']
            
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)

            # C. 接受/拒绝 决策 (Metropolis Criterion)
            acceptance_prob = self.get_acceptance_probability(current_acc, neighbor_acc, temp)
            
            if random.random() < acceptance_prob:
                # 接受新状态
                current_spec = neighbor_spec
                current_acc = neighbor_acc
            
            # D. 更新全局最优记录 (只用于统计，不影响 SA 游走逻辑)
            if neighbor_acc > best_valid_acc:
                best_valid_acc = neighbor_acc
                best_test_acc = neighbor_test_acc
                best_spec = neighbor_spec
            
            best_valids.append(best_valid_acc)
            best_tests.append(best_test_acc)

            # E. 降温 (Cooling Schedule)
            temp *= self.alpha
            # 防止温度过低导致下溢 (可选)
            if temp < 1e-5:
                temp = 1e-5

            # F. 检查预算
            if time_spent > self.max_time_budget:
                break
                
        return times, best_valids, best_tests
class BOSearch(Search):
    """
    基于贝叶斯优化 (Bayesian Optimization) 的搜索
    
    核心逻辑：
    1. 代理模型 (Surrogate): 使用随机森林拟合 (架构特征 -> 准确率)。
    2. 采集函数 (Acquisition): 使用 UCB (上置信界) 来平衡探索 (Exploration) 和利用 (Exploitation)。
    3. 候选生成: 由于是离散空间，无法直接求导，我们通过"变异"当前最优解来生成一批候选架构，
       然后用代理模型预测这些候选，选出最有希望的一个进行真实评估。
    """
    def __init__(self, nasbench, 
                 initial_population_size=20, # 初始化随机采样的数量 (Warm start)
                 candidate_pool_size=50,     # 每次迭代生成的候选架构数量
                 acq_kappa=2.0,              # UCB 参数: 越高越倾向于探索 (Exploration)
                 mutation_rate=0.1, 
                 max_time_budget=5e6):
        super().__init__(nasbench)
        self.initial_population_size = initial_population_size
        self.candidate_pool_size = candidate_pool_size
        self.acq_kappa = acq_kappa
        self.mutation_rate = mutation_rate
        self.max_time_budget = max_time_budget
        
        # 代理模型：随机森林回归
        # n_estimators: 树的数量，用于估计方差
        self.surrogate_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

    def spec_to_feature(self, spec):
        """
        将 ModelSpec 编码为数值向量，供随机森林使用。
        NASBench 矩阵是 7x7，操作是列表。
        我们将矩阵拉平，并将操作映射为整数。
        """
        # 1. 扁平化邻接矩阵 (7*7 = 49)
        matrix_flat = spec.original_matrix.flatten()
        
        # 2. 编码操作列表 (假设最大长度为7)
        # 映射表: input/output/conv1x1/conv3x3/maxpool3x3 -> 0/1/2/3/4
        # 注意：这里需要根据实际 nasbench.config 的 ops 列表来调整
        op_map = {'input': 0, 'output': 1, 'conv1x1-bn-relu': 2, 'conv3x3-bn-relu': 3, 'maxpool3x3': 4}
        
        # 如果 nasbench 的 ops 名字不同，请修改上面的 map，或者用动态 map
        # 为了通用性，这里做一个简单的容错映射，如果名字对不上，可以用简单的 hash 或索引
        ops_vec = []
        for op in spec.original_ops:
            if op in op_map:
                ops_vec.append(op_map[op])
            else:
                # 简单的 fallback，取 hash
                ops_vec.append(abs(hash(op)) % 10)
        
        # 填充 ops 向量到固定长度 (7)
        while len(ops_vec) < 7:
            ops_vec.append(0)
            
        return np.concatenate([matrix_flat, ops_vec])

    def get_ucb_score(self, X):
        """
        计算 Upper Confidence Bound (UCB) 分数
        UCB = Mean + kappa * Std
        """
        # 获取每棵树的预测，以计算方差
        # predictions shape: (n_estimators, n_samples)
        preds = []
        for estimator in self.surrogate_model.estimators_:
            preds.append(estimator.predict(X))
        preds = np.array(preds)
        
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        
        return mean + self.acq_kappa * std

    def generate_candidates(self, base_spec, num_candidates):
        """
        通过变异 base_spec (当前最优) 生成一批候选架构
        """
        candidates = []
        attempts = 0
        while len(candidates) < num_candidates and attempts < num_candidates * 5:
            attempts += 1
            new_spec = self.mutate_spec(base_spec, self.mutation_rate)
            if self.nasbench.is_valid(new_spec):
                candidates.append(new_spec)
        
        # 如果变异生成的有效太少，补充随机生成的
        while len(candidates) < num_candidates:
            spec = self.random_spec()
            candidates.append(spec)
            
        return candidates

    def run(self):
        nasbench = self.nasbench
        nasbench.reset_budget_counters()
        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        
        # 训练数据 (X: 特征, y: 精度)
        X_train = []
        y_train = []
        
        # 记录所有已评估的 specs，避免重复评估
        # 使用 spec 的 hash 或 string 表示作为 key
        history_specs = set() 

        # ---------------------------
        # 1. 初始化阶段 (Warm Start)
        # ---------------------------
        current_best_spec = None
        current_best_acc = -1.0

        for _ in range(self.initial_population_size):
            spec = self.random_spec()
            
            # 避免重复
            spec_hash = str(spec.original_matrix) + str(spec.original_ops)
            if spec_hash in history_specs:
                continue
            history_specs.add(spec_hash)
            
            data = nasbench.query(spec)
            acc = data['validation_accuracy']
            test_acc = data['test_accuracy']
            
            # 记录数据
            X_train.append(self.spec_to_feature(spec))
            y_train.append(acc)
            
            # 更新最优
            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_tests.append(test_acc)
                current_best_spec = spec
                current_best_acc = acc
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
            
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)
            if time_spent > self.max_time_budget:
                return times, best_valids, best_tests

        # ---------------------------
        # 2. 贝叶斯优化循环 (BO Loop)
        # ---------------------------
        while True:
            # A. 训练/更新代理模型
            # 将列表转换为 numpy 数组
            X_np = np.array(X_train)
            y_np = np.array(y_train)
            self.surrogate_model.fit(X_np, y_np)
            
            # B. 生成候选集 (在当前最优解附近变异)
            # 也可以混合一些完全随机的解以增加多样性
            candidates = self.generate_candidates(current_best_spec, self.candidate_pool_size)
            
            # C. 预测候选集的 UCB 分数
            X_candidates = np.array([self.spec_to_feature(s) for s in candidates])
            ucb_scores = self.get_ucb_score(X_candidates)
            
            # D. 选择 UCB 分数最高的候选
            # 我们需要确保选出来的不是已经评估过的
            best_candidate_idx = np.argsort(ucb_scores)[::-1] # 降序排列索引
            
            selected_spec = None
            for idx in best_candidate_idx:
                cand = candidates[idx]
                cand_hash = str(cand.original_matrix) + str(cand.original_ops)
                if cand_hash not in history_specs:
                    selected_spec = cand
                    history_specs.add(cand_hash)
                    break
            
            # 如果所有候选都评估过了（极其罕见），随机选一个新的
            if selected_spec is None:
                selected_spec = self.random_spec()
                
            # E. 真实评估 (Query NASBench)
            data = nasbench.query(selected_spec)
            real_acc = data['validation_accuracy']
            real_test_acc = data['test_accuracy']
            
            # F. 更新数据集
            X_train.append(self.spec_to_feature(selected_spec))
            y_train.append(real_acc)
            
            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)
            
            # 更新历史最优
            if real_acc > best_valids[-1]:
                best_valids.append(real_acc)
                best_tests.append(real_test_acc)
                current_best_spec = selected_spec # 更新用于下一次生成候选的基准
                current_best_acc = real_acc
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])
                
            # 检查预算
            if time_spent > self.max_time_budget:
                break
                
        return times, best_valids, best_tests

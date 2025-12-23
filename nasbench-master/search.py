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
    def __init__(self, nasbench, max_time_budget=5e6,
                 return_mode="trajectory", max_evals=150):
        """
        return_mode:
            - "trajectory": 原始行为（times, best_valids, best_tests）
            - "final_test_error": 仅返回论文里的 test error（150 evals）
        """
        super().__init__(nasbench)
        self.max_time_budget = max_time_budget
        self.return_mode = return_mode
        self.max_evals = max_evals

    def run(self):
        nasbench = self.nasbench
        nasbench.reset_budget_counters()

        times, best_valids, best_tests = [0.0], [0.0], [0.0]

        eval_count = 0  

        while True:
            #限制150用来公平评估
            if self.return_mode == "final_test_error":
                if eval_count >= self.max_evals:
                    break

            spec = self.random_spec()
            data = nasbench.query(spec)
            eval_count += 1  
            if data['validation_accuracy'] > best_valids[-1]:
                best_valids.append(data['validation_accuracy'])
                best_tests.append(data['test_accuracy'])
            else:
                best_valids.append(best_valids[-1])
                best_tests.append(best_tests[-1])

            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)

            if time_spent > self.max_time_budget:
                break

        if self.return_mode == "final_test_error":
            final_test_acc = best_tests[-1]
            final_test_error = 1.0 - final_test_acc
            return final_test_error

        return times, best_valids, best_tests


class evolutionSearch(Search):
    """
    基于进化算法的搜索
    """
    def __init__(self, nasbench,
                 population_size=100,
                 tournament_size=10,
                 crossover_rate=0.3,
                 mutation_rate=0.9,
                 max_time_budget=5e6,
                 return_mode="trajectory",
                 max_evals=150):
        super().__init__(nasbench)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_time_budget = max_time_budget
        self.return_mode = return_mode
        self.max_evals = max_evals

    def crossover_spec(self, spec1, spec2, crossover_rate):
        """邻接矩阵按行随机 + 操作向量按位置随机交叉"""
        if random.random() > crossover_rate:
            return copy.deepcopy(random.choice([spec1, spec2]))
        while True:
            mat1 = spec1.original_matrix
            mat2 = spec2.original_matrix
            new_mat = np.zeros_like(mat1)
            for i in range(mat1.shape[0]):
                new_mat[i] = mat1[i] if random.random() < 0.5 else mat2[i]

            ops1 = spec1.original_ops
            ops2 = spec2.original_ops
            new_ops = []
            for i in range(len(ops1)):
                if i == 0:
                    new_ops.append(ops1[0])
                elif i == len(ops1) - 1:
                    new_ops.append(ops1[-1])
                else:
                    new_ops.append(ops1[i] if random.random() < 0.5 else ops2[i])

            new_spec = api.ModelSpec(new_mat, new_ops)
            if self.nasbench.is_valid(new_spec):
                return new_spec

    def run(self, crossover=True):
        """
        trajectory 模式：time budget
        final_test_error 模式：150 architecture evaluations
        """
        nasbench = self.nasbench
        nasbench.reset_budget_counters()

        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        population = []
        eval_count = 0  

        # 种群初始化
        for _ in range(self.population_size):
            if self.return_mode == "final_test_error" and eval_count >= self.max_evals:
                break

            spec = self.random_spec()
            data = nasbench.query(spec)
            eval_count += 1

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

        while True:
            if self.return_mode == "final_test_error" and eval_count >= self.max_evals:
                break

            while True:
                if crossover:
                    sample = self.random_combination(population, self.tournament_size)
                    par1 = sorted(sample, key=lambda i: i[0])[-1][1]
                    sample = self.random_combination(population, self.tournament_size)
                    par2 = sorted(sample, key=lambda i: i[0])[-1][1]
                    child_spec = self.crossover_spec(par1, par2, self.crossover_rate)
                    new_spec = self.mutate_spec(child_spec, self.mutation_rate)
                else:
                    sample = self.random_combination(population, self.tournament_size)
                    parent = sorted(sample, key=lambda i: i[0])[-1][1]
                    new_spec = self.mutate_spec(parent, self.mutation_rate)

                if nasbench.is_valid(new_spec):
                    break

            data = nasbench.query(new_spec)
            eval_count += 1

            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)

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

        if self.return_mode == "final_test_error":
            final_test_acc = best_tests[-1]
            return 1.0 - final_test_acc

        return times, best_valids, best_tests

class SASearch(Search):
    """
    基于模拟退火 (Simulated Annealing) 的搜索
    """
    def __init__(self, nasbench,
                 initial_temp=0.1,
                 final_temp=0.001,
                 alpha=0.99,
                 mutation_rate=0.1,
                 max_time_budget=5e6,
                 return_mode="trajectory",
                 max_evals=150):
        super().__init__(nasbench)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.mutation_rate = mutation_rate
        self.max_time_budget = max_time_budget
        self.return_mode = return_mode
        self.max_evals = max_evals

    def get_acceptance_probability(self, acc_current, acc_new, temp):
        if acc_new > acc_current:
            return 1.0
        return math.exp((acc_new - acc_current) / temp)

    def run(self):
        nasbench = self.nasbench
        nasbench.reset_budget_counters()

        times, best_valids, best_tests = [0.0], [0.0], [0.0]
        eval_count = 0 

        # 1. 初始化
        current_spec = self.random_spec()
        data = nasbench.query(current_spec)
        eval_count += 1

        current_acc = data['validation_accuracy']
        best_spec = current_spec
        best_valid_acc = current_acc
        best_test_acc = data['test_accuracy']

        best_valids.append(best_valid_acc)
        best_tests.append(best_test_acc)

        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        temp = self.initial_temp

        # 2. 退火循环
        while True:
            if self.return_mode == "final_test_error" and eval_count >= self.max_evals:
                break

            # A. 生成邻居
            while True:
                neighbor_spec = self.mutate_spec(current_spec, self.mutation_rate)
                if nasbench.is_valid(neighbor_spec):
                    break

            # B. 评估邻居
            data = nasbench.query(neighbor_spec)
            eval_count += 1

            neighbor_acc = data['validation_accuracy']
            neighbor_test_acc = data['test_accuracy']

            time_spent, _ = nasbench.get_budget_counters()
            times.append(time_spent)

            # C. Metropolis 接受准则
            if random.random() < self.get_acceptance_probability(
                current_acc, neighbor_acc, temp
            ):
                current_spec = neighbor_spec
                current_acc = neighbor_acc

            # D. 更新全局最优（仅用于统计）
            if neighbor_acc > best_valid_acc:
                best_valid_acc = neighbor_acc
                best_test_acc = neighbor_test_acc
                best_spec = neighbor_spec

            best_valids.append(best_valid_acc)
            best_tests.append(best_test_acc)

            # E. 降温
            temp *= self.alpha
            if temp < 1e-5:
                temp = 1e-5

            # F. 原始 time budget（完全保留）
            if time_spent > self.max_time_budget:
                break

    
        if self.return_mode == "final_test_error":
            return 1.0 - best_test_acc

        return times, best_valids, best_tests

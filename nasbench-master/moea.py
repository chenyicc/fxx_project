import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from search import evolutionSearch
class MOEASearch(evolutionSearch):
    """
    MOEA for NASBench: NSGA-II + External Pareto Archive (epsilon pruning)

    必选目标: maximize validation accuracy
    可选目标: minimize params, minimize training_time
    统一最小化形式: (-acc, params?, time?)
    """

    def __init__(self,
                 nasbench,
                 pop_size=100,
                 offspring_size=None,
                 crossover=True,
                 crossover_rate=0.3,
                 mutation_rate=0.9,
                 max_time_budget=5e6,
                 use_params=False,
                 use_time=False,
                 archive_max=300,
                 archive_eps=None,
                 field_map=None,
                 seed=None):

        # ✅ 正确：初始化 evolutionSearch（从而间接初始化 Search）
        super().__init__(
            nasbench=nasbench,
            population_size=pop_size,
            tournament_size=2,          # MOEA 不用它，但必须给
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_time_budget=max_time_budget
        )

        # ===== MOEA 自己的参数 =====
        self.pop_size = int(pop_size)
        self.offspring_size = int(offspring_size) if offspring_size else int(pop_size)
        self.crossover = bool(crossover)

        self.use_params = bool(use_params)
        self.use_time = bool(use_time)

        self.archive_max = int(archive_max)
        self.archive_eps = archive_eps
        self.field_map = field_map or {}

        self.cache = {}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ---------------------------
    # Utils: NASBench fields & cache
    # ---------------------------
    @staticmethod
    def _get_from_data(data, candidates, default=None):
        for k in candidates:
            if k in data:
                return data[k]
        return default

    @staticmethod
    def _spec_key(spec):
        mat = tuple(map(tuple, spec.original_matrix))
        ops = tuple(spec.original_ops)
        return (mat, ops)

    def _make_obj(self, val_acc, params, ttrain):
        obj = [-float(val_acc)]  # minimize -acc
        if self.use_params:
            obj.append(float(params))
        if self.use_time:
            obj.append(float(ttrain))
        return tuple(obj)

    def evaluate(self, spec):
        key = self._spec_key(spec)
        if key in self.cache:
            ind = copy.deepcopy(self.cache[key])
            ind["spec"] = spec
            return ind

        data = self.nasbench.query(spec)

        val_keys = self.field_map.get("val", ["validation_accuracy"])
        test_keys = self.field_map.get("test", ["test_accuracy"])
        par_keys = self.field_map.get("params", ["trainable_parameters", "parameters", "param_count"])
        time_keys = self.field_map.get("time", ["training_time", "train_time", "runtime"])

        val = self._get_from_data(data, val_keys, default=None)
        if val is None:
            raise KeyError(f"Cannot find val acc in data keys: {list(data.keys())}")

        test = self._get_from_data(data, test_keys, default=None)
        params = self._get_from_data(data, par_keys, default=None)
        ttrain = self._get_from_data(data, time_keys, default=None)

        if self.use_params and params is None:
            raise KeyError(f"use_params=True but params key not found. data keys: {list(data.keys())}")
        if self.use_time and ttrain is None:
            raise KeyError(f"use_time=True but time key not found. data keys: {list(data.keys())}")

        ind = {
            "spec": spec,
            "val": float(val),
            "test": float(test) if test is not None else None,
            "params": float(params) if params is not None else None,
            "ttrain": float(ttrain) if ttrain is not None else None,
            "obj": self._make_obj(val, params, ttrain),
            "rank": 0,
            "crowd": 0.0,
            "age": 0,
        }
        self.cache[key] = copy.deepcopy(ind)
        return ind

    # ---------------------------
    # NSGA-II core
    # ---------------------------
    @staticmethod
    def dominates(a, b):
        fa, fb = a["obj"], b["obj"]  # minimization
        return all(x <= y for x, y in zip(fa, fb)) and any(x < y for x, y in zip(fa, fb))

    def fast_nondom_sort(self, pop):
        S = {id(p): [] for p in pop}
        n = {id(p): 0 for p in pop}
        fronts = [[]]

        for p in pop:
            Sp = []
            np_ = 0
            for q in pop:
                if p is q:
                    continue
                if self.dominates(p, q):
                    Sp.append(q)
                elif self.dominates(q, p):
                    np_ += 1
            S[id(p)] = Sp
            n[id(p)] = np_
            if np_ == 0:
                p["rank"] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[id(p)]:
                    n[id(q)] -= 1
                    if n[id(q)] == 0:
                        q["rank"] = i + 1
                        nxt.append(q)
            i += 1
            fronts.append(nxt)

        fronts.pop()
        return fronts

    @staticmethod
    def crowding_distance(front):
        if not front:
            return
        m = len(front[0]["obj"])
        for p in front:
            p["crowd"] = 0.0

        for k in range(m):
            front.sort(key=lambda x: x["obj"][k])
            front[0]["crowd"] = front[-1]["crowd"] = float("inf")
            fmin = front[0]["obj"][k]
            fmax = front[-1]["obj"][k]
            if fmax == fmin:
                continue
            for i in range(1, len(front) - 1):
                prevv = front[i - 1]["obj"][k]
                nextv = front[i + 1]["obj"][k]
                front[i]["crowd"] += (nextv - prevv) / (fmax - fmin)

    def environmental_select(self, pop, k):
        fronts = self.fast_nondom_sort(pop)
        new_pop = []
        for f in fronts:
            self.crowding_distance(f)
            if len(new_pop) + len(f) <= k:
                new_pop.extend(f)
            else:
                f.sort(key=lambda x: x["crowd"], reverse=True)
                new_pop.extend(f[: (k - len(new_pop))])
                break
        return new_pop

    @staticmethod
    def tournament_select(pop):
        a, b = random.sample(pop, 2)
        ka = (a["rank"], -a["crowd"])
        kb = (b["rank"], -b["crowd"])
        return a if ka < kb else b

    # ---------------------------
    # External Archive (Pareto + epsilon grid pruning)
    # ---------------------------
    def nondominated_filter(self, pop):
        nd = []
        for p in pop:
            if not any((q is not p) and self.dominates(q, p) for q in pop):
                nd.append(p)
        return nd

    def epsilon_prune(self, archive):
        if self.archive_eps is None or len(archive) <= self.archive_max:
            return archive[: self.archive_max]

        eps = tuple(self.archive_eps)
        m = len(archive[0]["obj"])
        if len(eps) != m:
            return archive[: self.archive_max]

        buckets = {}
        for p in archive:
            key = []
            for i, v in enumerate(p["obj"]):
                e = eps[i]
                key.append(int(math.floor(v / e)) if e > 0 else int(v))
            key = tuple(key)

            if key not in buckets:
                buckets[key] = p
            else:
                cur = buckets[key]
                # 优先 rank 更小；rank相同时 crowd 更大
                kp = (p.get("rank", 0), -p.get("crowd", 0.0))
                kc = (cur.get("rank", 0), -cur.get("crowd", 0.0))
                if kp < kc:
                    buckets[key] = p

        pruned = list(buckets.values())
        pruned = self.environmental_select(pruned, min(len(pruned), self.archive_max))
        return pruned

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        nb = self.nasbench
        nb.reset_budget_counters()

        history = {"times": [], "best_acc": [], "knee": []}

        # 默认 eps（你可按数据尺度调整）
        if self.archive_eps is None:
            eps = [1e-4]  # -acc
            if self.use_params:
                eps.append(1e5)   # params
            if self.use_time:
                eps.append(1e-2)  # time
            self.archive_eps = tuple(eps)

        def knee_point(front):
            if not front:
                return None
            F = np.array([p["obj"] for p in front], dtype=float)
            mn = F.min(axis=0)
            mx = F.max(axis=0)
            denom = np.where(mx > mn, mx - mn, 1.0)
            Fn = (F - mn) / denom
            idx = int(np.argmin(np.linalg.norm(Fn, axis=1)))
            return front[idx]

        # 1) init population
        population = []
        while len(population) < self.pop_size:
            spec = self.random_spec()
            if not nb.is_valid(spec):
                continue
            population.append(self.evaluate(spec))
            t, _ = nb.get_budget_counters()
            if t > self.max_time_budget:
                break

        population = self.environmental_select(population, min(len(population), self.pop_size))
        archive = self.nondominated_filter(population)
        archive = self.environmental_select(archive, min(len(archive), self.archive_max))
        archive = self.epsilon_prune(archive)

        # 2) evolve
        while True:
            offspring = []
            while len(offspring) < self.offspring_size:
                p1 = self.tournament_select(population)["spec"]
                if self.crossover:
                    p2 = self.tournament_select(population)["spec"]
                    child = self.crossover_spec(p1, p2, self.crossover_rate)
                    new_spec = self.mutate_spec(child, self.mutation_rate)
                else:
                    new_spec = self.mutate_spec(p1, self.mutation_rate)

                if not nb.is_valid(new_spec):
                    continue
                offspring.append(self.evaluate(new_spec))

                t, _ = nb.get_budget_counters()
                if t > self.max_time_budget:
                    break

            R = population + offspring
            population = self.environmental_select(R, self.pop_size)

            # update archive
            nd_pop = self.nondominated_filter(population)
            archive = self.nondominated_filter(archive + nd_pop)
            archive = self.environmental_select(archive, min(len(archive), self.archive_max))
            archive = self.epsilon_prune(archive)

            # logging
            t, _ = nb.get_budget_counters()
            best = max(population, key=lambda x: x["val"])
            front0 = self.nondominated_filter(population)
            knee = knee_point(front0)

            history["times"].append(t)
            history["best_acc"].append(best["val"])
            history["knee"].append(knee)

            if t > self.max_time_budget:
                break

        archive = self.nondominated_filter(archive)
        return history, archive, population

    # ---------------------------
    # Plot Pareto front / projections
    # ---------------------------
    @staticmethod
    def plot_pareto(front, use_params=False, use_time=False, title="Pareto Front", show_knee=True):
        if not front:
            print("Empty front.")
            return

        acc = np.array([p["val"] for p in front], dtype=float)
        params = np.array([p["params"] for p in front], dtype=float) if use_params else None
        ttrain = np.array([p["ttrain"] for p in front], dtype=float) if use_time else None

        def knee_idx(front_):
            F = np.array([p["obj"] for p in front_], dtype=float)
            mn = F.min(axis=0); mx = F.max(axis=0)
            denom = np.where(mx > mn, mx - mn, 1.0)
            Fn = (F - mn) / denom
            return int(np.argmin(np.linalg.norm(Fn, axis=1)))

        k = knee_idx(front) if show_knee else None

        if (not use_params) and (not use_time):
            plt.figure()
            plt.scatter(np.arange(len(acc)), acc)
            plt.xlabel("solution index")
            plt.ylabel("validation accuracy")
            plt.title(title)
            plt.grid(True)
            plt.show()
            return

        if use_params ^ use_time:
            plt.figure()
            if use_params:
                plt.scatter(params, acc)
                if show_knee:
                    plt.scatter(params[k], acc[k], marker="x")
                plt.xlabel("params (minimize)")
            else:
                plt.scatter(ttrain, acc)
                if show_knee:
                    plt.scatter(ttrain[k], acc[k], marker="x")
                plt.xlabel("training time (minimize)")
            plt.ylabel("validation accuracy (maximize)")
            plt.title(title)
            plt.grid(True)
            plt.show()
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(params, ttrain, acc)
        if show_knee:
            ax.scatter(params[k], ttrain[k], acc[k], marker="x")
        ax.set_xlabel("params (min)")
        ax.set_ylabel("train time (min)")
        ax.set_zlabel("val acc (max)")
        ax.set_title(title)
        plt.show()

        # 2D projections
        plt.figure()
        plt.scatter(params, acc)
        if show_knee:
            plt.scatter(params[k], acc[k], marker="x")
        plt.xlabel("params (min)"); plt.ylabel("val acc (max)")
        plt.title(title + " (params vs acc)")
        plt.grid(True); plt.show()

        plt.figure()
        plt.scatter(ttrain, acc)
        if show_knee:
            plt.scatter(ttrain[k], acc[k], marker="x")
        plt.xlabel("time (min)"); plt.ylabel("val acc (max)")
        plt.title(title + " (time vs acc)")
        plt.grid(True); plt.show()

        plt.figure()
        plt.scatter(params, ttrain)
        if show_knee:
            plt.scatter(params[k], ttrain[k], marker="x")
        plt.xlabel("params (min)"); plt.ylabel("time (min)")
        plt.title(title + " (params vs time)")
        plt.grid(True); plt.show()

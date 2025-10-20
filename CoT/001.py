import numpy as np
from typing import List, Tuple
import pandas as pd


class CoTConsensus:
    def __init__(self, node_count: int, representative_count: int, convergence_threshold: float = 1e-6):
        """
        初始化CoT共识算法

        Args:
            node_count: 全网节点总数 n
            representative_count: 要选取的代表节点数量 k
            convergence_threshold: 收敛阈值 ε
        """
        self.n = node_count
        self.k = representative_count
        self.epsilon = convergence_threshold
        self.trust_matrix = None
        self.iteration_history = []  # 存储迭代历史

    def initialize_trust_matrix(self, direct_trust_matrix: np.ndarray):
        """
        初始化并归一化信任矩阵

        Args:
            direct_trust_matrix: 直接信任矩阵 D (n x n)
        """
        # 确保对角线为0（节点对自身的信任度为0）
        np.fill_diagonal(direct_trust_matrix, 0)

        # 归一化处理：每行元素和为1
        row_sums = direct_trust_matrix.sum(axis=1, keepdims=True)
        # 避免除零错误，将零和的行设为均匀分布
        row_sums[row_sums == 0] = 1
        self.trust_matrix = direct_trust_matrix / row_sums

        print("归一化后的信任矩阵 C:")
        print(self.trust_matrix)

    def calculate_global_trust(self, max_iterations: int = 100) -> Tuple[np.ndarray, int]:
        """
        计算全网节点的全局信任值

        Args:
            max_iterations: 最大迭代次数

        Returns:
            tuple: (最终信任值向量, 实际迭代次数)
        """
        if self.trust_matrix is None:
            raise ValueError("请先初始化信任矩阵")

        # 初始化信任向量：所有节点平等
        trust_vector = np.ones(self.n) / self.n
        self.iteration_history = [trust_vector.copy()]  # 记录初始状态

        print(f"初始信任向量 T0: {trust_vector}")

        # 迭代计算
        for iteration in range(1, max_iterations + 1):
            # T_k = C^T · T_{k-1}
            new_trust_vector = self.trust_matrix.T @ trust_vector

            # 计算变化量
            delta = np.linalg.norm(new_trust_vector - trust_vector)

            print(f"迭代 {iteration}: T{iteration} = {new_trust_vector}, Δ = {delta:.6f}")

            # 记录迭代历史
            self.iteration_history.append(new_trust_vector.copy())

            # 检查是否收敛
            if delta < self.epsilon:
                print(f"在 {iteration} 次迭代后收敛")
                return new_trust_vector, iteration

            trust_vector = new_trust_vector

        print(f"达到最大迭代次数 {max_iterations}")
        return trust_vector, max_iterations

    def save_iteration_history_to_excel(self, filename: str = "cot_trust_iteration_history.xlsx",
                                        node_names: List[str] = None):
        """
        将迭代过程保存到Excel文件

        Args:
            filename: 保存的文件名
            node_names: 节点名称列表，如果为None则使用默认名称
        """
        if not self.iteration_history:
            print("没有迭代历史数据可保存")
            return

        if node_names is None:
            node_names = [f"节点{i}" for i in range(self.n)]

        # 创建DataFrame
        data = []
        for iteration, trust_vector in enumerate(self.iteration_history):
            row = {"迭代序号": iteration}
            for i, trust_value in enumerate(trust_vector):
                row[node_names[i]] = trust_value
            # 添加变化量（从第二次迭代开始）
            if iteration > 0:
                delta = np.linalg.norm(trust_vector - self.iteration_history[iteration - 1])
                row["变化量Δ"] = delta
            else:
                row["变化量Δ"] = 0.0
            data.append(row)

        df = pd.DataFrame(data)

        # 保存到Excel
        df.to_excel(filename, index=False, float_format="%.6f")
        print(f"\n迭代历史已保存到文件: {filename}")

        # 显示表格预览
        print("\n迭代历史预览:")
        print(df.head(min(13, len(df))))  # 显示前13行或全部行

    def select_representatives(self, trust_vector: np.ndarray) -> List[int]:
        """
        根据信任值选择代表节点

        Args:
            trust_vector: 全局信任值向量

        Returns:
            list: 代表节点索引列表
        """
        # 按信任值降序排序，获取索引
        sorted_indices = np.argsort(trust_vector)[::-1]

        # 选择前k个节点作为代表
        representatives = sorted_indices[:self.k].tolist()

        print("\n节点信任值排名:")
        for i, idx in enumerate(sorted_indices):
            rank = i + 1
            node_type = "★代表" if rank <= self.k else "普通"
            print(f"第{rank:2d}名: 节点{idx} - 信任值: {trust_vector[idx]:.6f} {node_type}")

        return representatives

    def run_consensus_step3(self, direct_trust_matrix: np.ndarray,
                            save_to_excel: bool = True,
                            excel_filename: str = "cot_trust_iteration_history.xlsx",
                            node_names: List[str] = None) -> List[int]:
        """
        运行完整的第三步共识过程

        Args:
            direct_trust_matrix: 直接信任矩阵 D
            save_to_excel: 是否保存迭代历史到Excel
            excel_filename: Excel文件名
            node_names: 节点名称列表

        Returns:
            list: 选出的代表节点列表
        """
        print("=" * 50)
        print("CoT共识算法 - 第三步：计算全局信任值并选举代表")
        print("=" * 50)

        # 1. 初始化信任矩阵
        self.initialize_trust_matrix(direct_trust_matrix)

        # 2. 计算全局信任值
        final_trust_vector, iterations = self.calculate_global_trust()

        # 3. 保存迭代历史到Excel
        if save_to_excel:
            self.save_iteration_history_to_excel(excel_filename, node_names)

        # 4. 选择代表节点
        representatives = self.select_representatives(final_trust_vector)

        print(f"\n🎯 最终选出的代表节点: {representatives}")
        return representatives


# 示例使用和测试
def create_example_trust_matrix():
    """
    创建示例信任矩阵（对应文档中的例子）
    """
    # 基于文档中的信任关系图创建直接信任矩阵 D
    # 节点: A(0), B(1), C(2), D(3)
    D = np.array([
        [0, 0.7, 0.5, 0.5],  # A对B,C,D的信任度
        [0.6, 0, 0.6, 0.1],  # B对A,C,D的信任度
        [0.5, 0.8, 0, 0.2],  # C对A,B,D的信任度
        [0.5, 0.5, 0.6, 0]  # D对A,B,C的信任度
    ])
    return D


def create_larger_example():
    """
    创建更大的示例（8个节点）
    """
    np.random.seed(42)  # 固定随机种子以便复现

    n = 8
    D = np.zeros((n, n))

    # 生成随机信任关系（大部分节点是诚实的）
    for i in range(n):
        for j in range(n):
            if i != j:
                # 诚实节点间信任度较高，恶意节点信任度较低
                if i < 6 and j < 6:  # 前6个是诚实节点
                    D[i, j] = np.random.uniform(0.6, 0.9)
                elif i >= 6 or j >= 6:  # 涉及恶意节点
                    D[i, j] = np.random.uniform(0.1, 0.4)

    # 设置恶意节点（节点6,7）之间的高互信（协同作弊）
    D[6, 7] = 0.9
    D[7, 6] = 0.9

    np.fill_diagonal(D, 0)
    return D


def compare_with_table_5_1():
    """
    与文档中的表5.1进行对比验证
    """
    print("=" * 60)
    print("与文档表5.1对比验证")
    print("=" * 60)

    # 使用文档中的信任矩阵
    D = np.array([
        [0, 0.7, 0.5, 0.5],
        [0.6, 0, 0.6, 0.1],
        [0.5, 0.8, 0, 0.2],
        [0.5, 0.5, 0.6, 0]
    ])

    # 使用文档中的节点名称
    node_names = ["T(A)", "T(B)", "T(C)", "T(D)"]

    cot = CoTConsensus(node_count=4, representative_count=2, convergence_threshold=1e-6)
    representatives = cot.run_consensus_step3(
        D,
        save_to_excel=True,
        excel_filename="table_5_1_comparison.xlsx",
        node_names=node_names
    )

    return cot


if __name__ == "__main__":
    print("CoT共识算法Python实现演示（含Excel导出）")
    print("=" * 50)

    # 测试1: 与文档表5.1对比
    cot_comparison = compare_with_table_5_1()

    # 测试2: 4节点例子
    print("\n" + "=" * 50)
    print("测试2: 4节点示例（自定义名称）")
    D_small = create_example_trust_matrix()
    node_names_small = ["节点A", "节点B", "节点C", "节点D"]
    cot_small = CoTConsensus(node_count=4, representative_count=2)
    reps_small = cot_small.run_consensus_step3(
        D_small.copy(),
        excel_filename="4_nodes_example.xlsx",
        node_names=node_names_small
    )

    # 测试3: 更大的8节点例子
    print("\n" + "=" * 50)
    print("测试3: 8节点示例（包含恶意节点）")
    D_large = create_larger_example()
    node_names_large = [f"Node_{i}" for i in range(8)]
    cot_large = CoTConsensus(node_count=8, representative_count=3)
    reps_large = cot_large.run_consensus_step3(
        D_large.copy(),
        excel_filename="8_nodes_example.xlsx",
        node_names=node_names_large
    )

    # 验证信任矩阵的性质
    print("\n" + "=" * 50)
    print("信任矩阵性质验证:")
    print(f"信任矩阵形状: {cot_large.trust_matrix.shape}")
    print(f"每行和是否都为1: {np.allclose(cot_large.trust_matrix.sum(axis=1), 1.0)}")
    print(f"矩阵元素范围: [{cot_large.trust_matrix.min():.3f}, {cot_large.trust_matrix.max():.3f}]")
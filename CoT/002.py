import hashlib
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    PRE_PREPARE = "PRE_PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    CHANGE_VIEW = "CHANGE_VIEW"


@dataclass
class Block:
    height: int
    prev_hash: str
    transactions: List[str]
    creator_id: int

    @property
    def hash(self):
        data = f"{self.height}{self.prev_hash}{''.join(self.transactions)}{self.creator_id}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ConsensusMessage:
    msg_type: MessageType
    block_height: int
    view: int
    node_id: int
    block: Optional[Block] = None
    signature: str = ""


class CotConsensusNode:
    def __init__(self, node_id: int, is_byzantine: bool = False):
        self.node_id = node_id
        self.is_byzantine = is_byzantine
        self.trust_value = 0.0

        # 共识状态
        self.current_height = 1
        self.current_view = 0
        self.primary_id = 0

        # 消息日志
        self.pre_prepare_msgs = {}
        self.prepare_msgs = {}
        self.commit_msgs = {}

        # 网络引用
        self.network = None

    def set_network(self, network):
        self.network = network

    def is_primary(self):
        """检查自己是否为主节点"""
        n = len(self.network.nodes)
        return (self.current_height + self.current_view) % n == self.node_id

    def create_block(self) -> Block:
        """主节点创建区块"""
        prev_hash = "0" * 64  # 简化处理
        transactions = [f"tx_{self.current_height}_{i}" for i in range(3)]
        return Block(
            height=self.current_height,
            prev_hash=prev_hash,
            transactions=transactions,
            creator_id=self.node_id
        )

    def broadcast_message(self, msg: ConsensusMessage):
        """广播消息到网络"""
        if self.network:
            self.network.broadcast_message(msg, self.node_id)

    def start_consensus(self):
        """开始共识流程"""
        if self.is_primary():
            print(f"节点 {self.node_id} 是主节点，开始创建区块...")
            block = self.create_block()

            # 发送 Pre-Prepare 消息
            pre_prepare_msg = ConsensusMessage(
                msg_type=MessageType.PRE_PREPARE,
                block_height=self.current_height,
                view=self.current_view,
                node_id=self.node_id,
                block=block
            )
            self.broadcast_message(pre_prepare_msg)
            print(f"节点 {self.node_id} 广播 Pre-Prepare 消息")

    def receive_message(self, msg: ConsensusMessage):
        """处理接收到的消息"""
        if self.is_byzantine:
            # 拜占庭节点可能不响应或发送错误消息
            return

        key = (msg.block_height, msg.view)

        if msg.msg_type == MessageType.PRE_PREPARE:
            self.handle_pre_prepare(msg, key)
        elif msg.msg_type == MessageType.PREPARE:
            self.handle_prepare(msg, key)
        elif msg.msg_type == MessageType.COMMIT:
            self.handle_commit(msg, key)
        elif msg.msg_type == MessageType.CHANGE_VIEW:
            self.handle_change_view(msg)

    def handle_pre_prepare(self, msg: ConsensusMessage, key: tuple):
        """处理 Pre-Prepare 阶段"""
        if key not in self.pre_prepare_msgs:
            self.pre_prepare_msgs[key] = []

        self.pre_prepare_msgs[key].append(msg)

        # 验证区块有效性
        if self.validate_block(msg.block):
            # 发送 Prepare 消息
            prepare_msg = ConsensusMessage(
                msg_type=MessageType.PREPARE,
                block_height=msg.block_height,
                view=msg.view,
                node_id=self.node_id
            )
            self.broadcast_message(prepare_msg)
            print(f"节点 {self.node_id} 发送 Prepare 消息")

    def handle_prepare(self, msg: ConsensusMessage, key: tuple):
        """处理 Prepare 阶段"""
        if key not in self.prepare_msgs:
            self.prepare_msgs[key] = []

        self.prepare_msgs[key].append(msg)

        # 检查是否收到 2f 个 Prepare 消息
        n = len(self.network.nodes)
        f = (n - 1) // 3  # 最大容错节点数
        if len(self.prepare_msgs[key]) >= 2 * f:
            # 发送 Commit 消息
            commit_msg = ConsensusMessage(
                msg_type=MessageType.COMMIT,
                block_height=msg.block_height,
                view=msg.view,
                node_id=self.node_id
            )
            self.broadcast_message(commit_msg)
            print(f"节点 {self.node_id} 发送 Commit 消息")

    def handle_commit(self, msg: ConsensusMessage, key: tuple):
        """处理 Commit 阶段"""
        if key not in self.commit_msgs:
            self.commit_msgs[key] = []

        self.commit_msgs[key].append(msg)

        # 检查是否收到 2f 个 Commit 消息
        n = len(self.network.nodes)
        f = (n - 1) // 3
        if len(self.commit_msgs[key]) >= 2 * f:
            print(f"节点 {self.node_id} 达成共识！区块 {msg.block_height} 已确认")
            self.finalize_block()

    def handle_change_view(self, msg: ConsensusMessage):
        """处理视图更换"""
        # 简化实现：直接增加视图编号
        self.current_view += 1
        print(f"节点 {self.node_id} 切换到视图 {self.current_view}")

    def validate_block(self, block: Block) -> bool:
        """验证区块有效性（简化）"""
        if not block or not block.transactions:
            return False
        return True

    def finalize_block(self):
        """最终确认区块"""
        self.current_height += 1
        # 清空消息日志，准备下一轮共识
        self.pre_prepare_msgs.clear()
        self.prepare_msgs.clear()
        self.commit_msgs.clear()


class CotNetwork:
    def __init__(self):
        self.nodes: List[CotConsensusNode] = []

    def add_node(self, node: CotConsensusNode):
        node.set_network(self)
        self.nodes.append(node)

    def broadcast_message(self, msg: ConsensusMessage, sender_id: int):
        """模拟网络广播"""
        for node in self.nodes:
            if node.node_id != sender_id:  # 不发送给自己
                # 模拟网络延迟
                time.sleep(0.01)
                node.receive_message(msg)

    def start_consensus_round(self):
        """启动一轮共识"""
        print(f"\n=== 开始第 {self.nodes[0].current_height} 轮共识 ===")
        for node in self.nodes:
            node.start_consensus()


# 演示代码
def demo_cot_consensus():
    # 创建网络和节点（4个节点，容错 f=1）
    network = CotNetwork()

    # 创建节点：3个正常节点，1个拜占庭节点
    for i in range(3):
        network.add_node(CotConsensusNode(node_id=i))

    # 添加一个拜占庭节点
    network.add_node(CotConsensusNode(node_id=3, is_byzantine=True))

    print("CoT 共识网络初始化完成")
    print(f"节点数量: {len(network.nodes)}, 最大容错: {(len(network.nodes) - 1) // 3}")

    # 运行两轮共识演示
    for round in range(2):
        network.start_consensus_round()
        time.sleep(1)  # 等待共识完成


if __name__ == "__main__":
    demo_cot_consensus()
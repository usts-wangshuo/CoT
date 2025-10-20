# CoT
基于信任的区块链共识算法实现 - Trust-based Consensus Algorithm for Blockchain

# CoT - Consensus based on Trust

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/你的用户名/Cot)](https://github.com/你的用户名/Cot/issues)
[![GitHub Stars](https://img.shields.io/github/stars/你的用户名/Cot)](https://github.com/你的用户名/Cot/stargazers)

🌐 **基于信任的区块链共识算法实现** | **Trust-based Consensus Algorithm for Blockchain**

## 📖 项目简介

CoT (Consensus based on Trust) 是一种面向联盟链的高效共识算法，通过量化节点间的信任关系，动态选择高信誉节点参与共识，在保证拜占庭容错安全性的同时，显著提升了系统的可扩展性。

### ✨ 核心特性

- ✅ **基于行为信任** - 通过节点交互数据动态计算信任值
- ✅ **高可扩展性** - 使用代表机制，通信复杂度从 O(n²) 降至 O(k²)
- ✅ **拜占庭容错** - 支持 f < n/3 的恶意节点容错
- ✅ **无需代币** - 不依赖加密货币，适合联盟链场景
- ✅ **最终一致性** - 保证全网数据的最终一致

## 🏗 算法架构

### CoT 共识流程

1. **信任量化** - 监测节点交互行为，计算直接信任度
2. **信任图构建** - 生成全网信任关系图和信任矩阵  
3. **信任值计算** - 使用 PageRank 算法迭代计算全局信任值
4. **区块生成** - 代表节点运行 PBFT 类协议生成新区块

### 信任量化公式
t_ij = (g_ij + 1) / (g_ij + β × u_ij + 2)


其中：
- `g_ij`: 有效交互次数
- `u_ij`: 无效交互次数  
- `β`: 惩罚系数（β > 1）

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包：见 `requirements.txt`

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/usts-wangshuo/Cot.git
cd Cot

# 安装依赖
pip install -r requirements.txt


# EVA-MARL: Offline Pre-trained Multi-Agent Decision Transformer

## 项目简介
EVA-MARL 是一个基于多智能体强化学习的项目，结合了离线预训练和在线学习技术，旨在解决复杂的多智能体决策问题。项目主要使用了基于 Transformer 的模型框架，并应用于 StarCraft II 的多智能体微操场景。

## 项目结构

## 安装步骤
1. 克隆项目：
   ```bash
   git clone https://github.com/your-repo/EVA-MARL.git
   cd EVA-MARL
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   bash install_sc2.sh
   ```
3. 运行示例：
   ```bash
   python run_madt_sc2.py --map_name 3s5z --cuda_id 0
   ```
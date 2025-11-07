# Turn-Level Reward Shaping 实现说明

## 概述

本次修改为 AgentFlow 训练框架引入了 **turn-level reward shaping** 功能，通过折扣因子（discount factor）实现更平滑的奖励分配策略，提升多轮对话场景下的训练效果。

## 问题分析

### 原有问题

在原始实现中（`daemon.py:659, 724-733`）：

1. **奖励过于稀疏**：所有 turns 共享同一个 final_reward，且只在最后一个 token 位置分配奖励
2. **学习信号弱**：中间 turns 没有明确的奖励信号，导致梯度传播困难
3. **训练不稳定**：稀疏奖励导致训练过程波动大，收敛慢

### 改进方案

使用 **exponential discount factor** 从最后一个 turn 向前分配奖励：

```python
# 对于 n 个 turns：
# turn_{n-1}: final_reward × gamma^0 = final_reward
# turn_{n-2}: final_reward × gamma^1
# turn_{n-3}: final_reward × gamma^2
# ...
# turn_0: final_reward × gamma^(n-1)
```

**优点**：
- ✅ 越靠后的 turn 获得越多奖励（更直接贡献）
- ✅ 平滑的奖励分布，梯度信号更强
- ✅ 符合强化学习理论（TD learning）
- ✅ 可通过 config 灵活控制

## 修改文件

### 1. `train/rollout.py`

**修改位置**: Lines 124-139, 169-176, 368-406

**修改内容**:
- 添加参数 `reward_shaping_gamma: float = 0.99`
- 添加参数 `enable_reward_shaping: bool = True`
- 将参数添加到配置映射中，支持从 config.yaml 读取

```python
class Rollout(LitAgent):
    def __init__(
        self,
        ...
        reward_shaping_gamma: float = 0.99,
        enable_reward_shaping: bool = True,
    ):
        ...
        # Reward shaping parameters
        self.reward_shaping_gamma = reward_shaping_gamma
        self.enable_reward_shaping = enable_reward_shaping
```

### 2. `agentflow/verl/daemon.py`

**修改位置**: Lines 99-146, 662-670, 796-833

**核心修改**:

1. **添加初始化参数**（Lines 111-112, 145-146）:
```python
def __init__(
    self,
    ...
    reward_shaping_gamma=0.99,
    enable_reward_shaping=True,
):
    ...
    self.reward_shaping_gamma = reward_shaping_gamma
    self.enable_reward_shaping = enable_reward_shaping
```

2. **实现 turn-level reward 计算函数**（Lines 796-833）:
```python
def _compute_turn_level_rewards(self, final_reward: float, n_turns: int) -> list:
    """
    Compute turn-level rewards with discount factor for smoother reward distribution.

    Strategy:
        Uses exponential discounting from the final turn backwards:
        - Last turn (n-1): final_reward * gamma^0 = final_reward
        - Turn (n-2): final_reward * gamma^1
        - ...
    """
    if not self.enable_reward_shaping or n_turns <= 1:
        return [final_reward] * n_turns

    turn_rewards = []
    for turn_idx in range(n_turns):
        distance_from_end = n_turns - turn_idx - 1
        turn_reward = final_reward * (self.reward_shaping_gamma ** distance_from_end)
        turn_rewards.append(turn_reward)

    return turn_rewards
```

3. **在 `get_train_data_batch` 中应用**（Lines 662-670）:
```python
for rollout_id, sample_info in finished_id_to_sample_info.items():
    # Compute turn-level rewards for this rollout
    final_reward = sample_info["reward"]
    n_turns = len(sample_info["trace_list"])
    turn_level_rewards = self._compute_turn_level_rewards(final_reward, n_turns)

    for turn_index, trace in enumerate(sample_info["trace_list"]):
        # Use turn-specific reward instead of final reward for all turns
        reward_list.append(turn_level_rewards[turn_index])
        ...
```

### 3. `agentflow/verl/trainer.py`

**修改位置**: Lines 421-422

**修改内容**:
```python
self.agent_mode_daemon = AgentModeDaemon(
    ...
    reward_shaping_gamma=self.config.agentflow.get("reward_shaping_gamma", 0.99),
    enable_reward_shaping=self.config.agentflow.get("enable_reward_shaping", True),
)
```

### 4. `train/config.yaml`

**修改位置**: Lines 22-23, 27-28

**新增配置**:
```yaml
env:
  ...
  REWARD_SHAPING_GAMMA: 0.99  # Discount factor for turn-level reward shaping
  ENABLE_REWARD_SHAPING: True  # Enable turn-level reward shaping

python_args:
  agentflow.reward_shaping_gamma: '${REWARD_SHAPING_GAMMA}'
  agentflow.enable_reward_shaping: '${ENABLE_REWARD_SHAPING}'
  ...
```

## 配置参数说明

### `REWARD_SHAPING_GAMMA` (float, 默认: 0.99)

折扣因子，控制奖励分配的平滑程度：

- **0.95-0.99**（推荐）: 更多信用给最后的 turns，适合目标导向的任务
  - 0.99: 温和折扣，保留 ~91% 到第一个 turn（10 turns 场景）
  - 0.95: 中等折扣，保留 ~60% 到第一个 turn（10 turns 场景）

- **0.80-0.90**: 更均匀的奖励分布，适合所有 turns 都同等重要的场景
  - 0.90: 保留 ~35% 到第一个 turn（10 turns 场景）
  - 0.80: 保留 ~13% 到第一个 turn（10 turns 场景）

### `ENABLE_REWARD_SHAPING` (bool, 默认: True)

是否启用 turn-level reward shaping：

- **True**: 使用折扣因子分配奖励（推荐）
- **False**: 所有 turns 获得相同的 final_reward（原始行为）

## 使用示例

### 场景 1: 标准设置（推荐）

```yaml
env:
  REWARD_SHAPING_GAMMA: 0.99
  ENABLE_REWARD_SHAPING: True
```

**效果**（3 turns, final_reward=1.0）:
- Turn 0: 0.9801
- Turn 1: 0.9900
- Turn 2: 1.0000

### 场景 2: 更激进的折扣

```yaml
env:
  REWARD_SHAPING_GAMMA: 0.95
  ENABLE_REWARD_SHAPING: True
```

**效果**（3 turns, final_reward=1.0）:
- Turn 0: 0.9025
- Turn 1: 0.9500
- Turn 2: 1.0000

### 场景 3: 禁用 reward shaping

```yaml
env:
  ENABLE_REWARD_SHAPING: False
```

**效果**（3 turns, final_reward=1.0）:
- Turn 0: 1.0
- Turn 1: 1.0
- Turn 2: 1.0

## 验证测试

运行测试脚本验证实现：

```bash
cd /path/to/AgentFlow
python test_reward_shaping.py
```

测试涵盖：
- ✅ 标准 gamma 值（0.99）
- ✅ 单 turn 场景
- ✅ 禁用 reward shaping
- ✅ 不同 gamma 值（0.95）
- ✅ 负奖励
- ✅ 多 turns 场景（10 turns）

## 理论基础

本实现基于强化学习中的 **Temporal Difference (TD) Learning** 和 **Discounted Return** 概念：

1. **TD Learning**: 通过时序差分更新价值估计
2. **Discount Factor (γ)**: 平衡即时奖励和未来奖励的权重
3. **Reward Shaping**: 通过添加辅助奖励引导学习，不改变最优策略

数学表达：

```
R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-t}·r_n

对于我们的实现：
turn_reward[t] = final_reward × γ^{n-t-1}
```

## 兼容性

- ✅ **向后兼容**: 设置 `ENABLE_REWARD_SHAPING=False` 恢复原始行为
- ✅ **默认启用**: 新的训练默认使用 reward shaping（gamma=0.99）
- ✅ **灵活配置**: 通过 config.yaml 轻松调整参数

## 性能影响

- **计算开销**: 几乎可以忽略（每个 rollout 仅增加一次循环计算）
- **内存开销**: 无额外内存消耗
- **训练效果**: 预期提升收敛速度和稳定性

## 调试建议

1. **监控奖励分布**: 观察不同 turns 的奖励值是否合理
2. **对比实验**:
   - Baseline: `ENABLE_REWARD_SHAPING=False`
   - Variant 1: `GAMMA=0.99`
   - Variant 2: `GAMMA=0.95`
3. **检查 turn 数量**: 确保 turn 数量统计正确

## 相关文件

- `train/rollout.py`: 参数定义和配置读取
- `agentflow/verl/daemon.py`: 核心 reward shaping 逻辑
- `agentflow/verl/trainer.py`: 参数传递
- `train/config.yaml`: 配置文件
- `test_reward_shaping.py`: 单元测试
- `REWARD_SHAPING_README.md`: 本文档

## 参考文献

1. Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
2. Ng et al., "Policy Invariance Under Reward Transformations" (1999)
3. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

## 作者与维护

- **实现者**: Claude Code
- **创建日期**: 2025-10-18
- **版本**: 1.0

如有问题或建议，请联系项目维护者。

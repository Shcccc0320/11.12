import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import shap  # 导入 SHAP 库
from V2XEnvironment import V2XEnvironment  # 确保正确导入 V2XEnvironment

# 选择设备：如果有可用的 GPU，则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_stations, total_bandwidth):
        super(Actor, self).__init__()
        self.num_vehicles = num_vehicles  # 车辆数量
        self.num_stations = num_stations  # 基站数量
        self.total_bandwidth = total_bandwidth  # 总带宽

        # 公共层
        self.common_layer1 = nn.Linear(state_dim, 256)  # 第一层，全连接层
        self.norm1 = nn.LayerNorm(256)  # 第一层归一化
        self.common_layer2 = nn.Linear(256, 256)  # 第二层，全连接层
        self.norm2 = nn.LayerNorm(256)  # 第二层归一化

        # 基站选择（每个车辆的 logits）
        self.base_station_layer = nn.Linear(256, num_vehicles * num_stations)  # 输出基站选择的 logits

        # 带宽分配（连续动作）
        self.bandwidth_layer = nn.Linear(256, num_vehicles)  # 输出每个车辆的带宽分配

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始化为零

    def forward(self, state):
        x = torch.relu(self.norm1(self.common_layer1(state)))  # 前向传播，经过第一层和激活函数
        x = torch.relu(self.norm2(self.common_layer2(x)))  # 经过第二层和激活函数

        # 基站选择 logits
        bs_logits = self.base_station_layer(x)  # 输出 logits
        bs_logits = bs_logits.view(-1, self.num_vehicles, self.num_stations)  # 调整形状

        # 为了数值稳定性，减去最大值
        bs_logits = bs_logits - bs_logits.max(dim=-1, keepdim=True)[0]

        bs_probs = torch.softmax(bs_logits, dim=-1)  # 通过 softmax 得到概率分布

        # 带宽分配
        bandwidth_raw = self.bandwidth_layer(x)  # 原始带宽输出
        bandwidth = torch.relu(bandwidth_raw) + 1e-6  # 通过 ReLU 确保正值，防止为零

        return bs_probs, bandwidth  # 返回基站选择概率和带宽分配

# 创建包装器模型
class ActorWrapper(nn.Module):
    def __init__(self, actor_model):
        super(ActorWrapper, self).__init__()
        self.actor = actor_model

    def forward(self, state):
        bs_probs, bandwidth = self.actor(state)
        output = bs_probs.mean(dim=(1, 2))  # 对车辆和基站维度求平均
        output = output.unsqueeze(-1)  # 在最后一维增加一个维度，形状变为 [batch_size, 1]
        # print(f"ActorWrapper output shape: {output.shape}")  # 可选：打印输出形状
        return output


# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_stations):
        super(Critic, self).__init__()
        self.num_vehicles = num_vehicles  # 车辆数量
        self.num_stations = num_stations  # 基站数量

        # 状态输入层
        self.state_layer = nn.Linear(state_dim, 256)  # 状态输入层
        self.norm1 = nn.LayerNorm(256)  # 状态层归一化

        # 动作输入层
        action_input_dim = num_vehicles * (num_stations + 1)  # 动作输入维度
        self.action_layer = nn.Linear(action_input_dim, 256)  # 动作输入层
        self.norm2 = nn.LayerNorm(256)  # 动作层归一化

        # 合并层
        self.common_layer1 = nn.Linear(512, 256)  # 合并状态和动作后的第一层
        self.norm3 = nn.LayerNorm(256)  # 合并层归一化
        self.common_layer2 = nn.Linear(256, 256)  # 合并后的第二层
        self.norm4 = nn.LayerNorm(256)  # 归一化
        self.output_layer = nn.Linear(256, 1)  # 输出层，输出 Q 值

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始化为零

    def forward(self, state, bs_action_one_hot, bandwidth_action):
        # 状态特征
        state_feature = torch.relu(self.norm1(self.state_layer(state)))  # 状态特征提取

        # 动作特征
        bs_action_flat = bs_action_one_hot.view(state.shape[0], -1)  # 展平基站选择动作
        bandwidth_action_flat = bandwidth_action.view(state.shape[0], -1)  # 展平带宽分配动作
        action_input = torch.cat([bs_action_flat, bandwidth_action_flat], dim=1)  # 合并动作输入
        action_feature = torch.relu(self.norm2(self.action_layer(action_input)))  # 动作特征提取

        # 合并状态和动作特征
        x = torch.cat([state_feature, action_feature], dim=1)  # 合并特征
        x = torch.relu(self.norm3(self.common_layer1(x)))  # 合并后的第一层
        x = torch.relu(self.norm4(self.common_layer2(x)))  # 合并后的第二层
        q_value = self.output_layer(x)  # 输出 Q 值
        return q_value  # 返回 Q 值

# DDPG Agent 定义
class DDPGAgent:
    def __init__(self, state_dim, num_vehicles, num_stations, total_bandwidth):
        self.num_vehicles = num_vehicles  # 车辆数量
        self.num_stations = num_stations  # 基站数量
        self.total_bandwidth = total_bandwidth  # 总带宽

        # 初始化 Actor 网络和目标网络
        self.actor = Actor(state_dim, num_vehicles, num_stations, total_bandwidth).to(device)  # 主网络
        self.actor_target = Actor(state_dim, num_vehicles, num_stations, total_bandwidth).to(device)  # 目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())  # 同步参数
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)  # 优化器

        # 初始化 Critic 网络和目标网络
        self.critic = Critic(state_dim, num_vehicles, num_stations).to(device)  # 主网络
        self.critic_target = Critic(state_dim, num_vehicles, num_stations).to(device)  # 目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())  # 同步参数
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)  # 优化器

        self.replay_buffer = deque(maxlen=100000)  # 经验回放缓存
        self.batch_size = 64  # 批量大小
        self.discount = 0.99  # 折扣因子
        self.tau = 0.01  # 软更新参数

        # ε-贪心探索参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率

        # 初始化 SHAP 解释器（可选）
        self.explainer = None  # 当需要时再初始化

    def select_action(self, state, exploration=True):
        if np.isnan(state).any() or np.isinf(state).any():
            print("State contains NaN or Inf:", state)  # 检查状态中的 NaN 或 Inf
            raise ValueError("State contains NaN or Inf")

        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)  # 将状态转换为张量
        bs_probs, bandwidth_raw = self.actor(state_tensor)  # 通过 Actor 网络获得动作

        if torch.isnan(bs_probs).any() or torch.isinf(bs_probs).any():
            print("bs_probs contains NaN or Inf:", bs_probs)  # 检查概率中的 NaN 或 Inf
            raise ValueError("bs_probs contains NaN or Inf")

        bs_probs_np = bs_probs.cpu().data.numpy().squeeze()  # 转换为 NumPy 数组
        bandwidth_np = bandwidth_raw.cpu().data.numpy().squeeze()  # 带宽分配
        bandwidth_np = np.maximum(bandwidth_np, 1e-6)  # 确保为正值

        # 基站选择
        if exploration and np.random.rand() < self.epsilon:
            # 随机选择基站（探索）
            base_station_selection = np.random.randint(0, self.num_stations, size=self.num_vehicles)
        else:
            # 贪心选择（利用）
            base_station_selection = np.argmax(bs_probs_np, axis=-1)

        # 规范化带宽分配
        bandwidth_allocations = bandwidth_np / np.sum(bandwidth_np)  # 归一化到和为1

        # 确保每个车辆的最小带宽分配
        min_bandwidth_fraction = 0.05  # 可根据需要调整
        bandwidth_allocations = np.maximum(bandwidth_allocations, min_bandwidth_fraction)

        # 重新归一化
        bandwidth_allocations = bandwidth_allocations / np.sum(bandwidth_allocations)

        # 缩放到总带宽
        bandwidth_allocations = bandwidth_allocations * self.total_bandwidth

        action = (base_station_selection, bandwidth_allocations)  # 动作

        # 可选：计算 SHAP 值（仅在特定情况下，如每100个 episode）
        if self.explainer is not None and np.random.rand() < 0.01:
            shap_values = self.explainer.shap_values(state_tensor)
            shap_values = shap_values[0]  # 提取第一个输出的 SHAP 值

            # 确保 shap_values 和 state_tensor 形状匹配
            shap_values = shap_values.reshape(1, -1)
            state_np = state_tensor.cpu().numpy()

            # 打印形状以进行验证
            print("shap_values shape:", shap_values.shape)
            print("state_np shape:", state_np.shape)



        return action  # 返回选择的动作

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0  # 经验不足时不训练

        batch = random.sample(self.replay_buffer, self.batch_size)  # 随机采样批量数据
        state, action, reward, next_state, done = zip(*batch)  # 解压批量数据

        state = torch.FloatTensor(np.array(state)).to(device)  # 转换为张量
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)  # 奖励
        next_state = torch.FloatTensor(np.array(next_state)).to(device)  # 下一状态
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)  # 终止标志

        # 处理动作
        base_station_selection = np.array([a[0] for a in action])  # 基站选择
        bandwidth_allocations = np.array([a[1] for a in action])  # 带宽分配

        # 将基站选择转换为独热编码
        bs_action_one_hot = np.zeros((self.batch_size, self.num_vehicles, self.num_stations))
        for i in range(self.batch_size):
            for v in range(self.num_vehicles):
                bs_action_one_hot[i, v, base_station_selection[i][v]] = 1

        bs_action_one_hot = torch.FloatTensor(bs_action_one_hot).to(device)  # 转换为张量
        bandwidth_action = torch.FloatTensor(bandwidth_allocations).to(device)  # 带宽动作张量

        # Critic 网络训练
        with torch.no_grad():
            next_bs_probs, next_bandwidth_raw = self.actor_target(next_state)  # 通过目标 Actor 网络预测下一个动作
            next_bandwidth = torch.relu(next_bandwidth_raw) + 1e-6  # 带宽分配
            next_bandwidth_np = next_bandwidth.cpu().data.numpy()

            # 规范化带宽分配
            next_bandwidth_allocations_np = next_bandwidth_np / np.sum(next_bandwidth_np, axis=1, keepdims=True)
            next_bandwidth_allocations_np = np.maximum(next_bandwidth_allocations_np, 0.05)
            next_bandwidth_allocations_np = next_bandwidth_allocations_np / np.sum(next_bandwidth_allocations_np, axis=1, keepdims=True)
            next_bandwidth_allocations_np = next_bandwidth_allocations_np * self.total_bandwidth
            next_bandwidth_allocations = torch.FloatTensor(next_bandwidth_allocations_np).to(device)

            # 贪心选择下一个基站
            next_bs_probs_np = next_bs_probs.cpu().data.numpy()
            next_base_station_selection = np.argmax(next_bs_probs_np, axis=-1)

            # 转换为独热编码
            next_bs_action_one_hot = np.zeros((self.batch_size, self.num_vehicles, self.num_stations))
            for i in range(self.batch_size):
                for v in range(self.num_vehicles):
                    next_bs_action_one_hot[i, v, next_base_station_selection[i][v]] = 1

            next_bs_action_one_hot = torch.FloatTensor(next_bs_action_one_hot).to(device)

            target_q = self.critic_target(next_state, next_bs_action_one_hot, next_bandwidth_allocations)  # 目标 Q 值
            target_q = reward + (1 - done) * self.discount * target_q  # 计算目标值

        current_q = self.critic(state, bs_action_one_hot, bandwidth_action)  # 当前 Q 值
        critic_loss = nn.MSELoss()(current_q, target_q)  # 计算 Critic 损失

        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print("Critic loss is NaN or Inf")  # 检查损失中的 NaN 或 Inf
            raise ValueError("Critic loss is NaN or Inf")

        self.critic_optimizer.zero_grad()  # 清零梯度
        critic_loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # 梯度裁剪
        self.critic_optimizer.step()  # 更新参数

        # Actor 网络训练
        bs_probs, bandwidth_raw = self.actor(state)  # 通过 Actor 网络预测动作
        bandwidth = torch.relu(bandwidth_raw) + 1e-6  # 带宽分配
        bandwidth_np = bandwidth.cpu().data.numpy()

        # 规范化带宽分配
        bandwidth_allocations_np = bandwidth_np / np.sum(bandwidth_np, axis=1, keepdims=True)
        bandwidth_allocations_np = np.maximum(bandwidth_allocations_np, 0.05)
        bandwidth_allocations_np = bandwidth_allocations_np / np.sum(bandwidth_allocations_np, axis=1, keepdims=True)
        bandwidth_allocations_np = bandwidth_allocations_np * self.total_bandwidth
        bandwidth_allocations = torch.FloatTensor(bandwidth_allocations_np).to(device)

        # 贪心选择基站
        bs_probs_np = bs_probs.cpu().data.numpy()
        base_station_selection = np.argmax(bs_probs_np, axis=-1)

        # 转换为独热编码
        bs_action_one_hot = np.zeros((self.batch_size, self.num_vehicles, self.num_stations))
        for i in range(self.batch_size):
            for v in range(self.num_vehicles):
                bs_action_one_hot[i, v, base_station_selection[i][v]] = 1
        bs_action_one_hot = torch.FloatTensor(bs_action_one_hot).to(device)

        actor_loss = -self.critic(state, bs_action_one_hot, bandwidth_allocations).mean()  # 计算 Actor 损失

        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("Actor loss is NaN or Inf")  # 检查损失中的 NaN 或 Inf
            raise ValueError("Actor loss is NaN or Inf")

        self.actor_optimizer.zero_grad()  # 清零梯度
        actor_loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 梯度裁剪
        self.actor_optimizer.step()  # 更新参数

        # 更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  # 软更新

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  # 软更新

        # 衰减 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 衰减探索率

        return actor_loss.item(), critic_loss.item()  # 返回损失值

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)  # 添加到经验回放缓存

    # 使用背景数据初始化 SHAP 解释器
    def initialize_shap_explainer(self, background_states):
        actor_wrapper = ActorWrapper(self.actor)
        self.explainer = shap.DeepExplainer(actor_wrapper, background_states)  # 初始化 SHAP 解释器


# 训练代理
env = V2XEnvironment()  # 创建环境
state_dim = env.observation_space.shape[0]  # 状态维度
num_vehicles = env.num_vehicles  # 车辆数量
num_stations = env.num_stations  # 基站数量
total_bandwidth = env.total_available_bandwidth  # 总带宽

agent = DDPGAgent(state_dim, num_vehicles, num_stations, total_bandwidth)  # 初始化代理

# 使用背景数据初始化 SHAP 解释器
# 收集一些背景状态
background_states = []
for _ in range(100):
    state = env.reset()
    background_states.append(state)
background_states = torch.FloatTensor(np.array(background_states)).to(device)
agent.initialize_shap_explainer(background_states)  # 初始化 SHAP 解释器

num_episodes = 5000  # 设定训练的总 episode 数
all_rewards = []  # 存储每个 episode 的总奖励
actor_losses = []  # 存储 Actor 的损失
critic_losses = []  # 存储 Critic 的损失

for episode in range(num_episodes):

    state = env.reset()  # 重置环境，获得初始状态
    if np.isnan(state).any() or np.isinf(state).any():
        print("Initial state contains NaN or Inf:", state)  # 检查初始状态中的 NaN 或 Inf
        raise ValueError("Initial state contains NaN or Inf")
    episode_reward = 0  # 初始化本 episode 的总奖励
    done = False  # 是否结束的标志
    step = 0  # 步数计数

    while not done:
        action = agent.select_action(state)  # 选择动作
        # 可选：记录动作
        # print(f"Episode {episode+1}, Step {step}, Action: {action}")

        next_state, reward, done, _ = env.step(action)  # 执行动作，获得下一个状态和奖励

        if np.isnan(reward) or np.isinf(reward):
            print("Reward contains NaN or Inf:", reward)  # 检查奖励中的 NaN 或 Inf
            raise ValueError("Reward contains NaN or Inf")

        agent.add_to_replay_buffer((state, action, reward, next_state, float(done)))  # 将经验添加到回放缓存
        actor_loss, critic_loss = agent.train()  # 训练代理
        state = next_state  # 更新状态
        episode_reward += reward  # 累加奖励
        step += 1  # 增加步数

    all_rewards.append(episode_reward)  # 记录本 episode 的总奖励
    actor_losses.append(actor_loss)  # 记录 Actor 损失
    critic_losses.append(critic_loss)  # 记录 Critic 损失
    print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")  # 输出信息

    # 每 1000 个 episode 绘制结果
    if (episode + 1) % 1000 == 0:
        plt.figure()
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DDPG on V2X Environment')
        plt.show()

        plt.figure()
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Losses')
        plt.legend()
        plt.show()



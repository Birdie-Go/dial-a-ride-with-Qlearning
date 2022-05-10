# coding = utf-8
import numpy as np
import random
from tqdm import tqdm
import math

class Qlearning():
    def __init__(self, G):
        '''
        初始化函数
        '''
        self.G = G
        np.random.seed(1) # 设置随机种子
        self.Q = np.random.randn(self.G.m, pow(3, self.G.n), self.G.m) # 随机初始化Q表,0-notyet,1-running,2-finish
    
    def next_step(self, x, state, toward):
        '''
        由当前状态(x, state)和运动方向toward得到下一个状态和reward
        '''
        nx = toward
        ns = state
        temp = state
        reward = 0
        for i in range(self.G.n):
            if temp % 3 == 0 and self.G.v[i] == toward:
                ns += pow(3, i)
                reward += 500
            if temp % 3 == 1 and self.G.v[i + self.G.n] == toward:
                ns += pow(3, i)
                reward += 1000
            temp //= 3
        reward += -self.G.t[x][nx] * 1000
        # print(f"{x} {nx} {state} {self.G.v}")
        return [nx, ns, reward]

    def next_toward(self, x, state, epsilon):
        '''
        ε-greedy策略选择下一个动作
        '''
        e = random.random()
        res = 0

        Q_max = self.Q[x][state][0]
        toward = 0
        for i in range(self.G.m):
            if self.Q[x][state][i] >= Q_max:
                Q_max = self.Q[x][state][i]
                toward = i
        res = toward

        # 有ε的概率随机选择一个动作
        if e < epsilon or toward == x:
            res = random.randint(0, self.G.m - 1)
            while res == x:
                res = random.randint(0, self.G.m - 1)
        return res

    def finish(self, state):
        '''
        判断是否终止
        '''
        if (state == pow(3, self.G.n) - 1):
            return False
        return True
    
    def forward(self, _runs = 10, _episodes = 100000, _epsilon = 0.2, _alpha = 0.1, _gama = 0.8):
        '''
        Q-Learning方法，迭代更新Q表
        '''
        runs = _runs
        episodes = _episodes
        epsilon = _epsilon
        alpha = _alpha
        gama = _gama
        rewards = np.zeros(episodes)
        print("Training Q-Learning...")
        for i in range(runs):
            # 执行episodes轮
            print(f"Start running {i}...")
            for episode in tqdm(range(episodes)):
                [x, state] = [self.G.s, 0]
                reward_sum = 0
                while self.finish(state):
                    # 采用ε-greedy策略选择下一步动作
                    toward = self.next_toward(x, state, epsilon)
                    [next_x, next_state, reward] = self.next_step(x, state, toward)
                    next_toward = self.next_toward(next_x, next_state, -1)
                    reward_sum += reward
                    # 更新Q表
                    self.Q[x][state][toward] += alpha * (reward + gama * self.Q[next_x][next_state][next_toward] - self.Q[x][state][toward])
                    # 更新状态
                    x = next_x
                    state = next_state
                reward_sum += -self.G.t[x][self.G.s] * 1000
                rewards[episode] += reward_sum / 1000
            self.print_optimal_way()
        # 平均平滑reward曲线
        rewards /= runs
        avg_rewards = []
        for i in range(499):
            avg_rewards.append(np.mean(rewards[ : i + 1]))
        for i in range(500, len(rewards) + 1):
            avg_rewards.append(np.mean(rewards[i - 500 : i]))
        return avg_rewards
    
    def print_optimal_way(self):
        '''
        打印最佳策略和路径
        '''
        strategy = np.zeros((self.G.m, pow(3, self.G.n)), dtype = int)
        # 根据Q表得到每一个状态的最佳策略
        for x in range(self.G.m):
            for state in range(pow(3, self.G.n)):
                toward = self.next_toward(x, state, -1)
                strategy[x][state] = toward
        
        # 从起点开始搜索,打印最佳路径
        print("--------optimal way of Q-Learning----------", end = '\n')
        print(self.G.p)
        print(self.G.v)
        [x, state] = [self.G.s, 0]
        cost = 0
        print(f"{x}", end = "")
        while self.finish(state):
            toward = self.next_toward(x, state, -1)
            [next_x, next_state, reward] = self.next_step(x, state, toward)
            print(f" -> {next_x}", end = "")
            cost += math.sqrt(pow(self.G.p[x][0] - self.G.p[next_x][0], 2) + pow(self.G.p[x][1] - self.G.p[next_x][1], 2))
            x = next_x
            state = next_state
        cost += math.sqrt(pow(self.G.p[x][0] - self.G.p[self.G.s][0], 2) + pow(self.G.p[x][1] - self.G.p[self.G.s][1], 2))
        print(f" -> {self.G.s}", end = "\n")
        print(f"The cost is {cost}")

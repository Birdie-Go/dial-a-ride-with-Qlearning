# coding = utf-8
from QLearning import *
from DQN import *
from graph import *
import matplotlib.pyplot as plt

# 设置graph
[m, n] = [30, 10]
Graph = graph(m, n)

# Q-Learning算法
model_Qlearning = Qlearning(Graph)
model_DQN = DQN(m)
Qlearning_rewards,DQN_rewards,RNN_rewards = model_Qlearning.forward()

# 绘制图像
plt.plot(range(len(Qlearning_rewards)), Qlearning_rewards, label = "Q-Learning")
plt.legend(loc = "lower right")
plt.show()

# 打印最佳路径
model_Qlearning.print_optimal_way()
print("--------optimal way of Q-Learning----------", end = '\n')
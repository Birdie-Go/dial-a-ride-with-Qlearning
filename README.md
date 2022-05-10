# DIAL-A-RIDE PROBLEM BASED ON REINFORCEMENT LEARNING

### 摘要

Reinforcement learning has recently shown the prospect of learning quality solutions in many combinatorial optimization problems. The solver based on actor-critical model shows high effectiveness in various routing problems, including traveling salesman problem (TSP). Unfortunately, in the Dial-a-Ride problem, due to many parameters and limitations, no scholars have used reinforcement learning method to solve the problem. Starting with the simple actor-critical model, this paper presents a framework for solving Dial-a-Ride problem based on reinforcement learning. Based on this model, it is easy to integrate deep reinforcement learning models, including attention-based end-to-end models.

强化学习最近在许多组合优化问题中显示出学习质量解决方案的前景。基于Actor-Critic模型的求解器在各种路由问题上表现出很高的有效性，包括旅行商问题（TSP）。不幸的是，在Dial-a-ride问题当中，由于问题的参数和限制比较多，目前还没有学者采用强化学习的方法对该问题进行求解。本文从简单的Actor-Critic模型入手，给出了基于强化学习求解Dial-a-ride问题的框架。在该模型的基础上，很容易能够融合深度强化学习模型，包括基于注意力的端到端模型。

关键词：DIAL-A-RIDE；Actor-Critic；Q-Learning；DQN；Attention

### 1 介绍

如今我们对通过手机App等方式打车比较习以为常了，但在智能手机未普及之前，人们通常是通过打电话来订车的，也就是Dial a ride。如果我们开了一家出租车公司，在接到这些顾客的服务请求之后，我们自然需要考虑一下怎么用我们手上的出租车来满足顾客的需求，在这样的背景下我们就需要考虑一下这个问题。

请求式的公共交通服务(也就是Dial a ride)首次尝试于1970年的美国俄亥俄州，1972年在英国的阿宾顿也有尝试。后来其可行性被证明以后相似的服务模式就开始在各地出现。

在早期，这种服务模式对年长者和由于残疾行动不便的人有着很大的便利。这一人群比较难正常地使用公共交通出行，因此这样的服务非常受这一人群的欢迎。美国在1990年的残疾人法案中，要求所有的公共交通代理公司为残疾人提供区别于普通的公共巴士服务的辅助客运系统（一般无固定路线或时间表），结果促进了DAR被广泛地引入、改进。

而在近些年，随着技术的发展（网络和移动通信、云计算、数据分析等）使得DAR能够以新的方式运行。如果大量的人单独使用私家车通勤的话，不仅会导致中央商务区和城市道路的拥堵，还会增加很多碳排放，滴滴的出现，使得这种服务模式又开始有市场了。

但是这种服务系统的运营是非常复杂的，在不同的应用场景下会有不同的特征，例如在医护领域会对时间窗约束的要求比较高，而对于残疾人则需要尽可能减少移动距离，有的运营公司会使用多车型的车队进行服务等等。这意味着需要有相应的算法进行实时性比较好的规划和排班才能够提供比较高质量的服务，这也是这一领域研究的意义。

Dial a ride是一个组合优化问题。组合优化问题类比为机器翻译过程(即序列到序列的映射), 神经网络的输入是问题的特征序列(如城市的坐标序列), 神经网络的输出是解序列(如城市的访问顺序), Vinyals 等人根据该思想, 对机器翻译领域的经典序列映射模型(Sequence-to-Sequence, Seq2Seq) 进行了改进, 提出了可以求解组合优化问题的指针网络模型 (Pointer Network,  Ptr-Net)[<sup>1</sup>](#refer-anchor-1) , 其原理详见第二章, 作者采用监督式学习的方式训练该网络并在TSP问题上取得了较好的优化效果. 多年来传统的组合优化算法都是以”迭代搜索”的方式进行求解, 但是 Vinyals 等人的模型可以利用神经网络直接输出问题解, 开启了组合优化一个新的研究领域。

路由问题是重要的组合优化问题，具有多种实际应用，包括送货和快递服务、拼车和物流。基于深度强化学习的方法也被证明可以很好地解决路由问题。在本文中，我们将强化学习引入Dial-a-ride问题，并设计了基于Actor-Critic模型的求解器，对模型采用马尔可夫决策进行建模，利用Q-Learning进行求解。在Q-Learning的基础上，提出了融合基于注意力编码器-LSTM 解码器混合模型的设想。

### 2 THE PROBLEM DEFINITION AND FORMULATION

We define Dial-a-ride formally and propose a Markov Decision Process (MDP) formulation.

#### 2.1 DIAL-A-RIDE DEFINED

Cordeau and Laporte曾经给这个问题下过定义：DAR问题即在满足顾客请求的前提下，在一个由所有节点和弧组成的完全图中组织车辆的行驶路线，使得这些行驶路线达到例如成本最低的规划目标。此外需要注意的是这种服务是共享的，即可能有多个不同的乘客同时在同一辆车上，跟我们说的顺风车有相似之处。

正如上述所说，在不同运营环境下的DAR会考虑不同的特征，一些比较常见的需要考虑的问题特征通常有以下几点：

1. 对需求节点的访问：如果不允许拒绝请求，则需要规划各个请求节点的访问；如果允许拒绝请求，则需要根据情况考虑接受哪些请求并作进一步规划。
2. 时间窗：每个顾客都能指定从出发点出发的时间窗和到达目的地的时间窗。
3. 车辆场站：即车辆一趟服务中开始服务与结束服务的地点。
4. 旅程：当车辆回到场站的时候视为完成了一次旅程。
5. 车辆容量：即车辆核载人数。
6. 乘行时间：乘客乘车时花费的时间。
7. 路线持续时间：车辆在一次旅程中所花费的时间。

通常在进行DAR的规划时需要在考虑上述特征的同时分配车辆，并为车辆作路径规划。我们知道，规划是需要一个规划目标的，规划目标可以从运营者视角出发（例如车辆行驶时间、总行驶距离、需要的车辆数量、司机的工作时间等）或者用户视角出发（例如乘行时间、等待时间、时间窗的满足情况等）。但是这两种视角对应的目标常常是冲突的。作为乘客，当然是希望能够尽可能地减少等待时间和乘行时间，但是这样就会造成运营成本的增加，比如需要增派车辆以达到这样的目标。

我们做出如下数学定义：

1. 有$k$辆容量不同的汽车，为一组包含起点和终点运输需求的顾客提供运输服务。

2. 定义节点集合$\mathcal{P}$，表示城市中地点的集合，其中
   $$
   |\mathcal{P}|=m
   $$

3. 定义需求的节点集合

$$
V=\{v_0,v_1,...,v_{2n}\}
$$

其中需求数为$n$，前$n$个节点为需求的起点、后$n$个节点为需求目的地。

4. 弧段集合

$$
E=\{(i,j)\mid i,j\in V,i\neq j\}
$$

5. 对于弧段$(i,j)$有运输成本$c_{i,j}^t$和旅行时间$t_{i,j}$。

6. 需求节点$i$具有出发时间窗$[e_i,l_i]$，和抵达时间窗$[e_{i+n},l_{i+n}]$，满足

$$
e_i<l_i\leq T
$$

7. 每个需求具有需求量$q_i$，车辆的容量为$Q$。
8. 当顾客$i$被车辆$j$拾取时，产生成本$c_{i,j}^P$。
9. 卡车返回车场的最晚时间$T$。

约束条件如下：

1. 每辆车从车场出发并最终返回，每个车辆访问每个弧段仅一次。
2. $v_i$必须在$v_{i+n}$之前。
3. 任一点的$q_i$不大于车辆容量。
4. 出发和抵达时间窗必须被满足。
5. 所有车辆在$T$之前返回。

目标函数为考虑运输成本和顾客偏好的成本最小。

为了让问题更加贴合实际网约车司机的应用需求，我们设定$k=1$。对于实际生活中，用于网约车的汽车通常是五座车，因此设定$Q=4$（计算司机在内）。为了简化问题， 我们将时间窗的约束调整至最宽，因此对于每一个需求结点，有$e_i=0,l_i=\inf$。在实际情况中，我们考虑的是小车行驶的速度，在小车速度恒定的情况下，旅行时间$t_{i,j}$正比于$\text{distance}(i,j)$。

#### 2.2 AN MDP FORMULATION OF DIAL-A-RIDE

为了在共享的城市环境中路由网约车，我们假设中央控制器观察整个地图信息并为网约车做出路由决策。我们将此问题形式化为具有离散步骤的马尔可夫决策过程（MDP）。

Markov Decision Process的状态用如下变量表示。首先，$O_t$表示第$t$个时刻的订单状态。状态表示为一个长度为$n$的三进制数，即
$$
O_t=(o^t_0o^t_1...o^t_{n-1})_3
$$
其中
$$
o^t_i\in\{0,1,2\}
$$
当$o^t_i=0$，表示第$i$份订单未被处理；当$o^t_i=1$，表示第$i$份订单的顾客已经在车上，但是为抵达目的地；当$o^t_i=2$，表示第$i$份订单的顾客已经抵达目的地，订单完成。接着用$x_t$表示汽车当前所在的点，其中
$$
x_t\in\mathcal{P}
$$
那么，有
$$
s_t:=(x_t,O_t)
$$
可以唯一标识一个状态。有初始状态
$$
s_0=(s,0)
$$
其中，$s$为车站，满足
$$
s\in\mathcal{P}
$$
由于图中每一个弧段只能被访问一次，实际上，这部分信息被隐式地包含在$s_t$当中。

在每一次运动过程中，考虑如何更新状态。假设当前状态为$s_t:=(x_t,O_t)$，中央处理器给出下一步小车将要到达的结点为$x_{t+1}$。考虑三种情况：

1）$x_{t+1}$不为需求上的结点，或者结点$y$上没有仍未完成的需求，那么订单状态不变，所在的位置需要改变
$$
s_{t+1}=(x_{t+1},O_t)
$$
2）目前车上有乘客的目的地为$y$，那么该乘客的订单状态设置为完成状态，即$\exist i,o_i^t=1\and v_{i+n}=x_{t+1}$，那么$O_{t+1}=O_t+3^{i-1}$。
$$
s_{t+1}=(x_{t+1},O_{t+1})
$$
3）有未执行的需求的起点在结点$x_{t+1}$上，那么该乘客的订单状态设置为运行状态，即$\exist i,o_i^t=0\and v_{i}=x_{t+1}$，那么$O_{t+1}=O_t+3^{i-1}$，那么
$$
s_{t+1}=(x_{t+1},O_{t+1})
$$
最终的结束时间为
$$
O_T=3^{n}-1
$$
即所有订单达到完成状态。最终的代价函数为
$$
C=\sum_{t=0}^Tt_{x,y}+x_T,s
$$
目标函数为
$$
\text{result}=\arg\min C
$$

### 3 MODEL

#### 3.1 Actor-Critic

从名字上看包括两部分，Actor和Critic。其中Actor使用策略函数，负责生成Action并和环境交互。而Critic使用价值函数，负责评估Actor的表现，并指导Actor下一阶段的动作。以下用符号$\pi$表示策略，$a$表示动作，$v$表示价值。

在Actor-Critic算法中，我们需要做两组近似，第一组是策略函数的近似：
$$
\pi_{\theta}(s,a)=P(a\mid s,\theta)\approx\pi(a\mid s)
$$
第二组是价值函数的近似，对于状态价值和动作价值函数分别是：
$$
\hat{v}(s,w)\approx v_{\pi}(s)
$$

$$
\hat{q}(s,a,w)\approx q_{\pi}(s,a)
$$

我们在蒙特卡罗策略梯度reinforce算法的基础上进行改造。

首先，在蒙特卡罗策略梯度reinforce算法中，我们的策略的参数更新公式是：
$$
\theta=\theta+\alpha \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right) v_{t}
$$
梯度更新部分中，$\nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right)$是分值函数，我们需要对$v_t$进行改进，使之从Critic中得到。

而对于Critic来说，用一个Q网络来做为Critic，这个Q网络的输入可以是状态，而输出是每个动作的价值或者最优动作的价值。

总体架构为，Critic通过Q网络计算状态的最优价值$v_t$, 而Actor利用$v_t$这个最优价值迭代更新策略函数的参数$\theta$,进而选择动作，并得到反馈和新的状态，Critic使用反馈和新的状态更新Q网络参数$w$，在后面Critic会使用新的网络参数$w$来帮Actor计算状态的最优价值$v_t$。

#### 3.2 Q-LEARNING MODEL

##### 3.2.1 时间差分方法

时间差分方法是一种估计值函数的方法，相较于蒙特卡洛使用完整序列进行更新，时间差分使用当前回报和下一时刻的价值进行估计，它直接从环境中采样观测数据进行迭代更新，时间差分方法学习的基本形式为：
$$
V(s) \leftarrow V(s)+\alpha[\gamma V(s')-V(s)]
$$

##### 3.2.2 ε-greedy策略

在做决策的时候，有$\epsilon$的概率随机选择未知的一个动作，剩下$1-\epsilon$的概率选择已有动过中动作价值最大的动作。理论上，最优策略应该一直采用后者，但事实上由于未探索其他动作，很有可能陷入局部最优。因此有$\epsilon$的概率随机选择未知的一个动作。

##### 3.2.3 Q-learning

Q-learning是一种时间差分算法，也是单步更新。其更新公式如下：
$$
Q(S,A) ← Q(S,A) + α*[R + γ*maxQ(S',a))-Q(s,a)]
$$
算法的步骤如下：

```
Initialize Q arbitrarily
Repeat (for each episode):
    Initialize S
    Repeat (for each step of episode):
        Choose a from s using policy derived from Q(ε-greedy)
        Take action a, observe r
        Q(S,A) ← Q(S,A) + α*[R + γ*maxQ(S',a))-Q(s,a)]
        S ← S'
    until S is terminal
```



#### 3.3 DQN MODEL

The Deep Q Network is hereinafter referred to as DQN.

DQN is a method that combines neural network and Q-learning. The traditional tabular form of Q-learning has such a bottleneck: we use a table to store each state and the Q value of each action corresponding to each state.

|         | action 1 | action 2 | ...  |
| ------- | -------- | -------- | ---- |
| state 1 | -2       | 2        |      |
| state 2 | -1       | 1        |      |
| state 3 | -2       | -5       |      |
| ...     |          |          |      |

However, in the problem of Dial-A-Ride, when the count of orders become large, the number of states is too large so that it may not be able to record in the table. We find out that the complexity of the orders state is a power of 3. The calculation of the problem is too large. If all tables are used for storage, the computer memory is far from enough. Moreover, the time complexity level is unimaginable every time you search for the corresponding state in such a large table. The neural network can solve this problem well. We can take the state and action as the input parameters of the neural network, and get the Q value of the action through the calculation of the neural network, so it is not necessary to record the Q value in the table. Neural network can be regarded as a function, which can actually be expressed in the form of Q(state, action). Then, in Q-learning, you also need to use this method: only input the current state, output all action values, and then select the action required in the next step according to the principle of $\epsilon-greedy$.

We use the formula to express how to update this network:

1. Approximate the action-value function
   $$
   \hat{q}(s,a,w)\approx q^{\pi}(s,a)
   $$

2. Minimize the MSE (mean-square error) between approximate action-value and true action-valude (assum oracle)
   $$
   J(w)=E_{\pi}[(q^{\pi}(s,a)-\hat{q}(s,a,w))^2]
   $$

3. Stochastic gradient descend to find a local minimum
   $$
   \triangle w=\alpha(q^{\pi}(s,a)-\hat{q}(s,a,w)\nabla_w\hat{q}(s,a,w))
   $$
   For Q-Learning ，the target is the TD target
   $$
   \triangle w=\alpha(R_{t+1}+\gamma \underset{a}max\ \hat{q}(s_{t+1},a,w)-\hat{q}(s_t,a_t,w))\nabla_w\hat{q}(s_t,a_t,w)
   $$



#### 3.4 端到端模型

实际上，对于解决路由模型，现阶段最好的近似模型为端到端模型。

端到端模型，即从输入端到输出端会得到一个预测结果，将预测结果和真实结果进行比较得到误差，将误差反向传播到网络的各个层之中，调整网络的权重和参数直到模型收敛或者达到预期的效果为止，中间所有的操作都包含在神经网络内部，不再分成多个模块处理。由原始数据输入，到结果输出，从输入端到输出端，中间的神经网络自成一体，这是端到端的。

编码器用于编码地图和订单信息。对于地图中的每个结点$k$，我们用一个长度为$2n+2$的向量$a^{(k)}$表示地图和订单的信息。其中，前两个维度表示结点的坐标信息，后$2n$个维度表示订单的信息。
$$
a_i^{(k)} = \begin{cases}  
x_i & i = 0 \\
y_i & i = 1 \\
0 & i\in[2,n+1],v_i\neq k\\
1 & i\in[2,n+1],v_i=k\\
0 & i\in[n+2,2n+1],v_i\neq k\\
1 & i\in[n+2,2n+1],v_i=k
\\
\end{cases}
$$
编码器采用多头注意力机制，详细介绍见讨论。



### 4 COMPUTATIONAL EXPERIMENTS

#### 4.1 TRAINING AND EVALUATION CONFIGURATIONS

##### 4.1.1 数据生成

生成数据采用在$[0,1]\times[0,1]$内均匀撒点的方式生成结点的坐标信息。

数据集中的图为一个完全图，$t_{i,j}$正比于两点之间的距离，采用
$$
t_{i,j}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
$$
订单的生成采用在结点中随机采样的方式，策略采用均匀采样。其中，采样的时候需要保证订单的起点和终点不是同一个结点。

##### 4.1.2 BASELINE

虽然精确性算法在小规模的算例上能够在可接受的时间内得到最优解，但是由于DAR问题是NP-Hard问题，所以研究的精力更多的是放在研究有效和高效的启发式算法上。从实际应用的角度出发，实际需求会需要能够更快响应的算法。

1. **Insertion Heuristics**
   这种方法比较简单，能够在较短的时间内找到可行解，但是解的质量不如元启发式算法。最早的应用由Jaw et al. (1986)完成。如今这种方法也被作为一种辅助用于找到可行解的方法在运用( Xiang et al., 2008; Wong et al., 2014; Markovi ´c et al., 2015 ),。
2. **Tabu Search**
   Cordeau and Laporte (2003)是最早提出在DAR问题中运用禁忌搜索算法的，他们使用了比较简单的邻域动作(将一个请求从一条路线转移到另一条路线)，有效果不错的多样化策略，例如对频繁移动这一行为进行惩罚，暂时接受不可行解。
3. **Simulated Annealing**
   在DAR问题上，SA并没有像其它方法那样被广泛的应用，少数几位作者( Mauri et al., 2009; Zidi et al., 2012; Reinhardt et al., 2013 )实现了标准的利用SA解DAR问题并取得了一定成果。
4. **Variable Neighborhood Search**
   Parragh et al. (2009)提出了第一个用于解决DAR问题的VNS算法，解决的是一个双目标的DAR问题。
5. **Large Neighborhood Search**
   在这个算法的研究上Ropke and Pisinger (2006)为带时间窗的接送问题设计的自适应大领域搜索算法为该算法在DAR问题上的应用打下了基础( Lehuédéet al., 2014;  Masmoudi et al., 2016; Molenbruch et al., 2017a )等人都曾为DAR问题设计LNS算法。

#### 4.3 RESULT OF DIAL-A-RIDE

##### 方法比较

下面的精确求解采用动态规划的方法进行求解。实验数据均为在10次实验下取平局值。Gap 为该方法的平均用时于所有方法的最快用时的差距。训练的episodes均为100000。

|            | 精确解用时(s) | Gap of Q-Learing | Gap of DQN | Gap of RNN |
| ---------- | ------------- | ---------------- | ---------- | ---------- |
| m=10,n=6   | 3.6632679336  | 0.00%            | 0.00%      | 0.00%      |
| m=30,n=10  | 7.6278214646  | 0.00%            | 0.00%      | 0.00%      |
| m=100,n=20 | -             | -                |            |            |

##### 价值函数图像





### 5 DISCUSSION

#### 5.1 编码器

对于给定输入的向量，计算三个值：queries, keys and values，其中
$$
\mathbf{q} \in \mathbb{R}^{d_{q}}, \mathbf{k} \in \mathbb{R}^{d_{q}}, \mathbf{v} \in \mathbb{R}^{d_{v}}
$$
输入$\mathbf{h}_{\mathbf{n}}$ ，是上一层网络的输出:
$$
\mathbf{q}_{n}=\mathbf{W}^{Q} \mathbf{h}_{n}, \quad \mathbf{k}_{n}=\mathbf{W}^{K} \mathbf{k}_{n}, \quad \mathbf{v}_{n}=\mathbf{W}^{V} \mathbf{h}_{n} \quad \forall n \in \mathcal{N},
$$
 $\mathbf{W}^{Q}, \mathbf{W}^{K}, \mathbf{W}^{V}$ 是学习量，大小为 $\left(d_{k} \times d_{h}\right),\left(d_{k} \times d_{h}\right)$,$\left(d_{v} \times d_{h}\right)$.

通过 queries 和 key 可以得到点之间的相关性
$$
u_{i, j}=\frac{\mathbf{q}_{i}^{\top} \mathbf{k}_{j}}{\sqrt{d_{k}}} \quad \forall i, j \in \mathcal{N} .
$$
通过相关性可以得到注意力权重$a_{i, j} \in[0,1]$ 
$$
a_{i, j}=\frac{e^{u_{i j}}}{\sum_{j^{\prime}} e^{u_{i, j^{\prime}}}} .
$$
节点 n 接受到的消息为
$$
\mathbf{h}_{n}^{\prime}=\sum_{j} a_{n, j} \mathbf{v}_{j} .
$$
假设头的数量是 $M$, 将$M$个头合并
$$
\mathbf{M H A}_{n}\left(\mathbf{h}_{1}, \ldots, \mathbf{h}_{N}\right)=\sum_{m=1}^{M} \mathbf{W}_{m}^{O} \mathbf{h}_{n m}^{\prime}
$$

然后进行批处理归一化

$$
\widehat{\mathbf{h}}_{n}^{l}=\mathbf{B} \mathbf{N}^{l}\left(\mathbf{h}_{n}^{l-1}+\mathbf{M H} \mathbf{A}_{n}\left(\mathbf{h}_{1}^{l-1}, \ldots, \mathbf{h}_{N}^{l-1}\right)\right)
$$
进入带有激活函数的全连接前馈网络
$$
\mathbf{h}_{n}^{l}=\mathbf{B} \mathbf{N}^{l}\left(\widehat{\mathbf{h}}_{n}^{l}+\mathbf{F} \mathbf{F}^{l}\left(\widehat{\mathbf{h}}_{n}^{l}\right)\right)
$$
where
$$
\left.\mathbf{F} \mathbf{F}^{l}\left(\widehat{\mathbf{h}}_{n}^{l}\right)\right)=\mathbf{W}^{\mathrm{ff}, 1} \cdot \operatorname{ReLU}\left(W^{\mathrm{ff}, 0} \widehat{\mathbf{h}}_{n}^{l}+\mathbf{b}^{\mathrm{ff}, 0}\right)+\mathbf{b}^{\mathrm{ff}, 1}
$$

#### 5.2 解码器

解码器和编码器的结构是一样的。解码器的隐状态输入$h^{(0)}$为编码器输出的地图和订单的压缩信息，而$x^{(i)}$为解码器解码出来的小车上一个状态的向量，$y^{(i)}$为当前单元输出的小车的下一个结点的概率向量，向量的维度为结点数量。

刚刚已经介绍，编码器输出的向量编码了整个输入序列 $x_{1}, \ldots, x_{T}$ 的信息。解码器通过将地图编码信息解码生成输出序列。给定训练样本中的输出序列 $y_{1}, y_{2}, \ldots, y_{T^{\prime}}$ ，对解码阶段每个时间步 $t^{\prime}$ ，解码器输出 $y_{t^{\prime}}$ 的条件概率将基于之前的输出序列 $y_{1}, \ldots, y_{t^{\prime}-1}$ 和地图信息，即 $\mathbb{P}\left(y_{t^{\prime}} \mid y_{1}, \ldots, y_{t^{\prime}-1}, \boldsymbol{c}\right)$ 。用下面的公式表示输出序列的联合概率函数:
$$
\mathbb{P}\left(y_{1}, \ldots, y_{T^{\prime}} \mid x_{1}, \ldots, x_{T}\right)=\prod_{t^{\prime}=1}^{T^{\prime}} \mathbb{P}\left(y_{t^{\prime}} \mid y_{1}, \ldots, y_{t^{\prime}-1}, \boldsymbol{c}\right)
$$
根据计算公式可以发现，之前的输出序列 $y_{1}, \ldots, y_{t^{\prime}-1}$ 和地图订单信息，即为上一个RNN单元传递过来的隐状态$h^{(i-1)}$。













### reference

<div id="refer-anchor-1"></div>[1]Vinyals O, Fortunato M, Jaitly N. Pointer networks.  In: Advances in Neural Information Processing  Systems, 2015


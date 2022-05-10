# coding = utf-8
import numpy as np
import math

class graph():
    '''
    初始化函数
    '''
    def __init__(self, _m, _n):
        self.m = _m
        self.k = 1
        self.s = 0
        self.Q = 4
        self.T = 1000
        self.p = np.random.rand(self.m, 2)
        self.t = np.zeros((self.m, self.m))
        self.ct = np.zeros((self.m, self.m))
        self.cp = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                self.t[i][j] = math.sqrt(pow(self.p[i][0] - self.p[j][0], 2) + pow(self.p[i][1] - self.p[j][1], 2))
        self.n = _n
        self.v = np.random.randint(0, self.m, 2 * self.n)
        for i in range(self.n):
            if self.v[i] == self.v[i + self.n]:
                self.v[i] = (self.v[i] + 1) % self.n
        self.e = np.zeros(2 * self.n)
        self.l = np.zeros(2 * self.n)
        self.l[:] += 1000
        self.q = np.zeros(2 * self.n, dtype = int)
        self.q[:] += 1
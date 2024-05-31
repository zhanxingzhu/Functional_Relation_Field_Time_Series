import numpy as np
import random

random.seed(0)
np.random.seed(0)

d = 40
t = 288
k = 7
T = np.random.rand(1, 2 ** k, 1)
T = T * 126.0 + 18.0
fi = np.random.rand(d, 2 ** k, 1)
fi = fi * 2.0 * np.pi
A = np.random.rand(1, 2 ** k, 1)
A = A * 70.0 + 30.0

Anoise = np.random.rand(d, 2 ** k, 1) * 0.1 + 0.95
noise = np.random.rand(d, 2 ** k, t) * 0.1 + 0.95
x = np.reshape(np.array(list(range(t))).astype(np.float64), [1, 1, t])
data = A * Anoise * (1.0 + np.sin(x / T * 2.0 * np.pi + fi) * 0.6) * noise

n0 = 0
n1 = 2 ** k
n2 = 0
n = 2 ** k
N = 2 ** (k + 1) - 1
data = np.concatenate([data, np.zeros([d, N - n, t])], axis=1)
adj = np.zeros([N, N])
for i in range(k - 1, -1, -1):
    for j in range(2 ** i):
        data[:, n1 + j] = np.sqrt(data[:, n0 + j * 2] * data[:, n0 + j * 2 + 1])
        adj[n1 + j, n0 + j * 2] = 1.0
        adj[n1 + j, n0 + j * 2 + 1] = 1.0
        adj[n0 + j * 2, n1 + j] = 1.0
        adj[n0 + j * 2 + 1, n1 + j] = 1.0
    n0 += 2 ** (i + 1)
    n1 += 2 ** i
    n2 += 2 ** i
data = np.reshape(np.transpose(data, [0, 2, 1]), [-1, 2 ** (k + 1) - 1, 1])
np.savez('data.npz', data=data, adj=adj)

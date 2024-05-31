import numpy as np
import torch


class reg_nonlinear:
    def __init__(self, device):
        self.res1 = np.load('../data/bintree/data.npz')['r1']
        self.res2 = np.load('../data/bintree/data.npz')['r2']
        self.res3 = np.load('../data/bintree/data.npz')['r3']
        self.res1 = torch.from_numpy(np.transpose(self.res1.astype(np.float32))).to(device)
        self.res2 = torch.from_numpy(np.transpose(self.res2.astype(np.float32))).to(device)
        self.res3 = torch.from_numpy(np.transpose(self.res3.astype(np.float32))).to(device)

    def __call__(self, xin, l2_=False, avg_=False, sum_=False):
        x = xin / (1e-6 + torch.norm(xin, p=2, dim=1, keepdim=True))
        r1 = torch.mm(x, self.res1)
        r2 = torch.mm(x, self.res2)
        r3 = torch.mm(x, self.res3)
        res = torch.abs(r1 * r2 - r3 ** 2)
        res_ = torch.sum(res, dim=1, keepdim=True)
        if l2_:
            return torch.mean(res ** 2)
        elif avg_:
            return torch.mean(res_)
        elif sum_:
            return torch.sum(res_)
        return res_

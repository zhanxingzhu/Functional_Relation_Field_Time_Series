import torch
import torch.nn as nn
import numpy as np
from model.layers import stblock, output_layer, output_layer_


class STGCN(nn.Module):
    def __init__(self, args, Llist, channels, device):
        super(STGCN, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.lag = args.lag
        self.Llist = torch.from_numpy(Llist.astype(np.float32)).to(device)
        self.default_graph = args.default_graph
        self.stb = nn.ModuleList()
        ko = self.lag
        for i in range(len(channels)):
            self.stb.append(stblock(self.num_nodes, args.ks, args.kt, channels[i], self.Llist, device))
            ko -= (args.kt * 2 - 2)
        if args.iter == 'iter':
            self.output_layer = output_layer(self.num_nodes, ko, channels[-1][-1], device)
        else:
            self.output_layer_ = output_layer_(self.num_nodes, ko, channels[-1][-1], device)
            self.end_conv = nn.Conv2d(1, args.horizon, kernel_size=(1, channels[-1][-1]), bias=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        self.reg_state = []
        for i in range(len(self.stb)):
            x, F = self.stb[i](x)
            self.reg_state.append(F + 0.0)
        if self.args.iter == 'iter':
            x = self.output_layer(x)
            output = x.permute(0, 2, 3, 1)
            return output
        else:
            x = self.output_layer_(x).permute(0, 2, 3, 1)
            if not x.is_contiguous():
                x = x.contiguous()
            output = self.end_conv(x)  # b, 1, n, c      b, t, n, 1
        return output

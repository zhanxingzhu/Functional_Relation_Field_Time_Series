import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


class ln(nn.Module):
    def __init__(self, N, C):
        super(ln, self).__init__()
        self.N = N
        self.C = C
        self.gamma = Parameter(torch.ones([1, self.C, 1, self.N], requires_grad=True))
        self.beta = Parameter(torch.zeros([1, self.C, 1, self.N], requires_grad=True))

    def forward(self, x):
        mean = torch.mean(x, dim=(1, 3), keepdim=True)
        std = torch.std(x, dim=(1, 3), keepdim=True)
        x = (x - mean) / (std + 1e-6) * self.gamma + self.beta
        return x


class tlayer(nn.Module):
    def __init__(self, kt, cin, cout, act, device):
        super(tlayer, self).__init__()
        self.cin = cin
        self.cout = cout
        self.act = act
        self.device = device
        self.kt = kt
        if cin > cout:
            self.linear = nn.Conv2d(cin, cout, kernel_size=1)
        if act == 'glu':
            self.tconv = nn.Conv2d(cin, cout*2, kernel_size=(kt, 1), padding=0)
            self.func = nn.Sigmoid()
        else:
            self.tconv = nn.Conv2d(cin, cout, kernel_size=(kt, 1), padding=0)
            if act == 'relu':
                self.func = nn.ReLU(inplace=True)
            elif act == 'sigmoid':
                self.func = nn.Sigmoid()

    def forward(self, x):
        [b, c, t, n] = list(x.size())
        if self.cin > self.cout:
            if not x.is_contiguous():
                x = x.contiguous()
            xin = self.linear(x)
        elif self.cin < self.cout:
            xin = torch.cat([x, torch.zeros([b, self.cout - self.cin, t, n]).to(self.device)], 1)
        else:
            xin = x + 0.0
        xin = xin[:, :, (self.kt-1):]
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.tconv(x)
        if self.act == 'glu':
            x = (x[:, :self.cout] + xin) * self.func(x[:, self.cout:])
        elif self.act == 'relu':
            x = self.func(x + xin)
        elif self.act == 'sigmoid':
            x = self.func(x)
        return x


class slayer(nn.Module):
    def __init__(self, ks, cin, cout, Llist, device):
        super(slayer, self).__init__()
        self.cin = cin
        self.cout = cout
        self.device = device
        self.Llist = Llist
        self.ks = ks
        if cin > cout:
            self.linear = nn.Conv2d(cin, cout, kernel_size=1)
        self.gconv = nn.Linear(cin * ks, cout)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        [b, c, t, n] = list(x.size())
        if self.cin > self.cout:
            if not x.is_contiguous():
                x = x.contiguous()
            xin = self.linear(x)
        elif self.cin < self.cout:
            xin = torch.cat([x, torch.zeros([b, self.cout - self.cin, t, n]).to(self.device)], 1)
        else:
            xin = x + 0.0
        xtmp = torch.mm(x.reshape(-1, n), self.Llist).reshape(b, c, t, self.ks, n).permute(0, 2, 4, 1, 3)
        xtmp = xtmp.reshape(-1, c * self.ks)
        xtmp = self.gconv(xtmp).reshape(b, t, n, self.cout).permute(0, 3, 1, 2)
        xtmp = self.relu(xtmp + xin)
        return xtmp


class stblock(nn.Module):
    def __init__(self, nodes, ks, kt, channels, Llist, device):
        super(stblock, self).__init__()
        self.tconv1 = tlayer(kt, channels[0], channels[1], 'glu', device)
        self.sconv = slayer(ks, channels[1], channels[1], Llist, device)
        self.tconv2 = tlayer(kt, channels[1], channels[2], 'relu', device)
        self.ln = ln(nodes, channels[2])

    def forward(self, x):
        x = self.tconv1(x)
        x = self.sconv(x)
        x = self.tconv2(x)
        x = self.ln(x)
        F = torch.mean(x, dim=(1, 2))
        return x, F


class output_layer(nn.Module):
    def __init__(self, nodes, kt, channels, device):
        super(output_layer, self).__init__()
        self.tconv1 = tlayer(kt, channels, channels, 'glu', device)
        self.ln = ln(nodes, channels)
        self.tconv2 = tlayer(1, channels, channels, 'sigmoid', device)
        self.tconv3 = tlayer(1, channels, 1, 'linear', device)

    def forward(self, x):
        x = self.tconv1(x)
        x = self.ln(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x


class output_layer_(nn.Module):
    def __init__(self, nodes, kt, channels, device):
        super(output_layer_, self).__init__()
        self.tconv = tlayer(kt, channels, channels, 'glu', device)

    def forward(self, x):
        x = self.tconv(x)
        return x


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

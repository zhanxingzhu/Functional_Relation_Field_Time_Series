import torch
import torch.nn as nn
from model.layers import AGCRNCell


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1, args=None):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        self.args = args
        # A = args.restrict  # .copy()
        # A = A[:, ::-1].copy()
        # n = np.shape(A)[1]
        # for k in range(n//2):
        #     A[:, (2*k):(2*k+2)] = A[:, (2*k):(2*k+2)][:, ::(-1)]
        # A = A.copy()
        # self.A = torch.from_numpy(A).to(args.device)
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        # reg_state = 0.0
        self.Ftilde = []
        pool = nn.AvgPool1d(64, stride=64)
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                if t == 0:
                    Ftilde = pool(state).permute(1, 0, 2).reshape(self.node_num, -1).permute(1, 0)
                else:
                    Ftilde += pool(state).permute(1, 0, 2).reshape(self.node_num, -1).permute(1, 0)
                # Ftilde = Ftilde.reshape(self.node_num, -1)
                # reg_state += torch.mean(torch.abs(torch.mm(self.A, Ftilde)) ** 2)
                inner_states.append(state)
            self.Ftilde.append(Ftilde / float(seq_length))
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        # self.restrictT = torch.from_numpy(args.restrict.transpose()).to(args.device)
        # A = args.restrict
        # self.AAT_invA = np.dot(np.linalg.inv(np.dot(A, A.transpose())), A)
        # self.AAT_invA = torch.from_numpy(self.AAT_invA).to(args.device)

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers, args)

        # predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      # B, T, N, hidden
        output = output[:, -1:, :, :]                                   # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))                         # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             # B, T, N, C
        self.reg_state = self.encoder.Ftilde
        return output

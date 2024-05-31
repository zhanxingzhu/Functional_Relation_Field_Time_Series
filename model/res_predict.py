import pandas as pd
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter


class predict(torch.nn.Module):
    def __init__(self, ind, device='cuda:0', unit=32):
        super().__init__()
        self.unit = unit
        self.m = np.shape(ind)[0]
        self.n = np.shape(ind)[1] - 1
        
        # self.param_1 = Parameter(torch.randn([self.m, self.n, unit], requires_grad=True) * 0.01)
        # self.param_2 = Parameter(torch.randn([self.m, unit, unit], requires_grad=True) * 0.01)
        # self.param_3 = Parameter(torch.randn([self.m, unit, unit], requires_grad=True) * 0.01)
        # self.param_4 = Parameter(torch.randn([self.m, unit, 1], requires_grad=True) * 0.01)
        
        # self.param_1 = Parameter(torch.zeros([self.m, self.n, unit], requires_grad=True))
        # self.param_2 = Parameter(torch.zeros([self.m, unit, unit], requires_grad=True))
        # self.param_3 = Parameter(torch.zeros([self.m, unit, unit], requires_grad=True))
        # self.param_4 = Parameter(torch.zeros([self.m, unit, 1], requires_grad=True))

        # add
        self.param_1 = Parameter(torch.empty([self.m, self.n, unit], requires_grad=True))
        self.param_2 = Parameter(torch.empty([self.m, unit, unit], requires_grad=True))
        self.param_3 = Parameter(torch.empty([self.m, unit, unit], requires_grad=True))
        self.param_4 = Parameter(torch.empty([self.m, unit, 1], requires_grad=True))
        self.relu_1 = torch.nn.LeakyReLU(inplace=True)
        self.relu_2 = torch.nn.LeakyReLU(inplace=True)
        self.relu_3 = torch.nn.LeakyReLU(inplace=True)
        self.relu_4 = torch.nn.LeakyReLU(inplace=True)
        self.gt_ind = torch.from_numpy(ind[:, 0]).to(device)
        self.input_ind = torch.reshape(torch.from_numpy(ind[:, 1:]), (-1,)).to(device)
        # self.dropout = nn.Dropout(p=0.5)
        # self.dropout = nn.Dropout(p=0.5)

        # add
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.param_1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.param_2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.param_3, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.param_4, a=math.sqrt(5))

    def forward(self, x):
        bs = list(x.size())[0]
        # '''
        param_1_ = self.param_1.repeat(bs, 1, 1)
        param_2_ = self.param_2.repeat(bs, 1, 1)
        param_3_ = self.param_3.repeat(bs, 1, 1)
        param_4_ = self.param_4.repeat(bs, 1, 1)
        # bias_1_ = self.bias_1.repeat(bs, 1, 1)
        # bias_2_ = self.bias_2.repeat(bs, 1, 1)
        # bias_3_ = self.bias_3.repeat(bs, 1, 1)
        # bias_4_ = self.bias_4.repeat(bs, 1, 1)
        self.gt = torch.index_select(x, 1, self.gt_ind)
        input_ = torch.reshape(torch.index_select(x, 1, self.input_ind), (-1, 1, self.n))
        # print('input_=', input_)
        y = self.relu_1(torch.bmm(input_, param_1_))  # + bias_1_
        y = self.relu_2(torch.bmm(y, param_2_))  # + bias_2_
        y = self.relu_3(torch.bmm(y, param_3_))  # + bias_3_
        y = self.relu_4(torch.bmm(y, param_4_))  # + bias_4_
        pred = torch.reshape(y, (-1, self.m))
        self.loss_item = torch.sum(torch.abs(pred - self.gt) ** 1, dim=1, keepdim=True)
        # self.loss = torch.sum(torch.abs(pred - self.gt) ** 2)
        self.loss = torch.sum(self.loss_item)
        return pred


def loss_func(pred, y, sum_=True):
    if not sum_:
        return torch.mean(torch.abs(pred - y), dim=0)
    loss = torch.mean(torch.sum(torch.abs(pred - y), dim=1))
    return loss


def mape_func(pred, y):
    loss = torch.mean(torch.abs(pred - y) / torch.abs(y + 1e-6))
    return loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default='bintree', type=str)
    parser.add_argument('--nodes_file', default='', type=str)
    parser.add_argument('--output_nodes_file', default='', type=str)
    parser.add_argument('--output_model_file', default='', type=str)
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--nodes', default=255, type=int)
    parser.add_argument('--thresh', default=1e-2, type=float)
    args = parser.parse_args()
    N = args.nodes
    ind = []
    for i in range(N):
        ind.append([i] + list(range(i)) + list(range(i+1, N)))
    ind = np.array(ind)
    K = args.K
    if len(args.nodes_file) > 0:
        ind = pd.read_csv(args.nodes_file, header=None).values
    print(ind)

    if args.dataset == 'bintree':
        data = np.load('../data/bintree/data.npz')['data'][:, :, 0].astype(np.float32)
        n = 30 * 288
        nte = 5 * 288
    elif args.dataset == 'miniapp2':
        data = np.load('../data/miniapp2/data.npz')['data'][:, :, 0].astype(np.float32)
        n = 18 * 288
        nte = 3 * 288
    else:
        data = np.load('../data/miniapp1/data.npz')['data'][:, :, 0].astype(np.float32)
        n = 15 * 288
        nte = 3 * 288
    # dmean = np.mean(data, axis=0, keepdims=True)
    # dstd = np.std(data, axis=0, keepdims=True)
    # data = (data - dmean) / dstd * 100.0
    dmean = np.mean(data, axis=0)
    dstd = np.std(data, axis=0)
    # print(dstd)
    print('mean:', np.mean(np.abs(data)))

    xtr = data[:n, :]
    xte = data[n:(n+nte), :]

    bs = 32
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = predict(ind=ind, device=device, nonzero_init=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)
    # '''
    for epoch in range(200):
        lst = list(range(n))
        random.shuffle(lst)
        xsh = xtr[np.array(lst), :]
        lmean = 0.0
        model.train()
        for i in range(n // bs):
            # mul = (np.random.rand(bs, 1) * 9.0 + 1.0).astype(np.float32)
            # mask = (np.random.rand(bs, 1) < 0.5).astype(np.float32)
            # mul = mask * mul + (1.0 - mask) / mul
            x_feed = torch.from_numpy(xsh[(i * bs):((i + 1) * bs), :].astype(np.float32)).to(device)
            pred = model(x_feed)
            # print(pred)
            loss = loss_func(pred, model.gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lmean += loss.item()
        print('train', epoch + 1, lmean / float(n // bs))
        model.eval()
        lmean = []
        mape_mean = 0.0
        for i in range(nte // bs):
            x_feed = torch.from_numpy(xte[(i * bs):((i + 1) * bs), :].astype(np.float32)).to(device)
            pred = model(x_feed)
            # print(pred)
            loss = loss_func(pred, model.gt, sum_=False)
            lmean.append(loss.cpu().detach().numpy())
            mape = mape_func(pred, model.gt)
            mape_mean += mape.item()
            # lmean += loss.item()
        lmean = np.stack(lmean, axis=0)
        print('test', epoch + 1, np.mean(np.sum(lmean, axis=1), axis=0), mape_mean / float(nte // bs))
    # print(np.mean(lmean, axis=0))
    # print(np.sqrt(np.mean(lmean)))
    if len(args.output_model_file) > 0:
        torch.save(model.state_dict(), args.output_model_file)
    # '''
    # model.load_state_dict(torch.load('constraints.pth'))
    lmean = []
    mape_mean = 0.0
    for i in range(n // bs):
        x_feed = torch.from_numpy(xtr[(i * bs):((i + 1) * bs), :].astype(np.float32)).to(device)
        x_feed = x_feed
        pred = model(x_feed)
        # print(pred)
        loss = loss_func(pred, model.gt, sum_=False)
        lmean.append(loss.cpu().detach().numpy())
        mape = mape_func(pred, model.gt)
        mape_mean += mape.item()
        # lmean += loss.item()
    lmean = np.stack(lmean, axis=0)
    print('train',  np.mean(np.sum(lmean, axis=1), axis=0), mape_mean / float(n // bs))
    print(np.mean(lmean, axis=0) / dmean[ind[:, 0]])
    lmean = np.mean(lmean, axis=0)
    print(np.sum((lmean < 0.05 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.04 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.03 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.02 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.01 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.005 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.002 * dmean[ind[:, 0]]).astype(np.float32)))
    lmean = []
    mape_mean = 0.0
    for i in range(nte // bs):
        x_feed = torch.from_numpy(xte[(i * bs):((i + 1) * bs), :].astype(np.float32)).to(device)
        x_feed = x_feed
        pred = model(x_feed)
        # print(pred)
        loss = loss_func(pred, model.gt, sum_=False)
        lmean.append(loss.cpu().detach().numpy())
        mape = mape_func(pred, model.gt)
        mape_mean += mape.item()
        # lmean += loss.item()
    lmean = np.stack(lmean, axis=0)
    print('validation',  np.mean(np.sum(lmean, axis=1), axis=0), mape_mean / float(nte // bs))
    print(np.mean(lmean, axis=0) / dmean[ind[:, 0]])
    lmean = np.mean(lmean, axis=0)
    print(np.sum((lmean < 0.05 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.04 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.03 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.02 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.01 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.005 * dmean[ind[:, 0]]).astype(np.float32)))
    print(np.sum((lmean < 0.002 * dmean[ind[:, 0]]).astype(np.float32)))
    grad = []
    if K < (np.shape(ind)[1] - 1):
        ind_ = []
        for i in range(np.shape(ind)[0]):
            model.zero_grad()
            # x_feed_ = xtr[:bs, :] + 0.0
            for j in range(nte // bs):
                x_feed_ = xte[(j*bs):(j*bs+bs)] + 0.0
                x_feed_ = torch.from_numpy(x_feed_.astype(np.float32)).to(device)
                x_feed = x_feed_.requires_grad_()
                x_feed.retain_grad()
                # x_feed.zero_grad()
                pred = model(x_feed)
                loss = torch.sum(pred[:, i])
                loss.backward()
                if j == 0:
                    g = np.linalg.norm(x_feed.grad.detach().cpu().numpy(), axis=0)
                else:
                    g += np.linalg.norm(x_feed.grad.detach().cpu().numpy(), axis=0)
            indmax = np.argsort(-g)
            # print(i, g)
            indmax = indmax[:K]
            ind_.append([ind[i, 0]] + list(indmax))
        ind_ = np.array(ind_)
    else:
        ind_ = ind
    if len(args.output_nodes_file) > 0:
        f = open(args.output_nodes_file, 'w')
        for i in range(np.shape(ind)[0]):
            if lmean[i] / dmean[ind[i, 0]] < args.thresh:
                print(','.join(map(str, list(ind_[i]))))
                f.writelines(','.join(map(str, list(ind_[i])))+'\n')
        f.close()

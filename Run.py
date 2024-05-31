import os
import sys

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
# from learn_res import rest
from torch.nn.parameter import Parameter
from model.res_predict import predict
import pandas as pd
from lib.graph_math import scaled_laplacian, cheb_poly_approx


Mode = 'train'
DEBUG = 'True'
# DATASET = 'PEMSD4'      # PEMSD4 or PEMSD8
DATASET = 'bintree'
DEVICE = 'cuda:0'


# parser
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--dataset', default='bintree', type=str)
parser.add_argument('--model', default='AGCRN', type=str)
parser.add_argument('--iter', default='non', type=str)
parser.add_argument('--constraints', default='learn', type=str)
parser.add_argument('--constraints_nodes', default='', type=str)
parser.add_argument('--constraints_net', default='', type=str)
parser.add_argument('--reg_lambda', default=0.01, type=float)
parser.add_argument('--resloss_lambda', default=0.01, type=float)
parser.add_argument('--proj_times', default=5, type=int)
parser.add_argument('--step_len', default=1.0, type=float)
args = parser.parse_args()
MODEL = args.model
DATASET = args.dataset
if MODEL == 'AGCRN':
    from model.AGCRN import AGCRN as Network
else:
    from model.STGCN import STGCN as Network
# get configuration
config_file = './conf/{}_{}.conf'.format(DATASET, MODEL)
# print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
for k in config.keys():
    print(k)
print(config.keys())
parser.add_argument('--mode', default=Mode, type=str)
parser.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
parser.add_argument('--debug', default=DEBUG, type=eval)
parser.add_argument('--cuda', default=True, type=bool)
# data
parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
parser.add_argument('--lag', default=config['data']['lag'], type=int)
parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
parser.add_argument('--tod', default=config['data']['tod'], type=eval)
parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
# model
if args.model == 'AGCRN':
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
else:
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--ks', default=config['model']['ks'], type=int)
    parser.add_argument('--kt', default=config['model']['kt'], type=int)
# train
parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
parser.add_argument('--seed', default=config['train']['seed'], type=int)
parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
parser.add_argument('--teacher_forcing', default=False, type=bool)
# parser.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
parser.add_argument('--real_value', default=config['train']['real_value'], type=eval,
                    help='use real value for loss calculation')
# test
parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
# log
parser.add_argument('--log_dir', default='./', type=str)
parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
parser.add_argument('--plot', default=config['log']['plot'], type=eval)
args = parser.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

# *************************************************************************#
if args.constraints_nodes == '':
    args.constraints_nodes = './constraints_model/' + DATASET + '_nodes.py'
if args.constraints_net == '':
    args.constraints_net = './constraints_model/' + DATASET + '_cnet.pth'
ind = pd.read_csv(args.constraints_nodes, header=None)
ind = ind.values
    
# adj matrix
if args.model == 'STGCN':
    datafilename = './data/' + args.dataset + '/' + 'data.npz'
    adj = np.load(datafilename)['adj']
    N = adj.shape[0]
    
    # true for frf enhanced constraint graph
    # false for backbone network
    use_constraint_graph = True
    
    if use_constraint_graph:
        adj = np.zeros((N, N))
        adj[range(N), range(N)] = 1
        for col_id in range(1, ind.shape[1]):
            ind0, ind1 = ind[:, 0], ind[:, col_id]
            adj[ind0, ind1] = 1
            adj[ind1, ind0] = 1
            
    adj = scaled_laplacian(adj).astype(np.float32)
    print('adj.shape=', adj.shape)
    Llist = cheb_poly_approx(adj, args.ks, args.num_nodes)


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


# init model
if MODEL == 'STGCN':
    model = Network(args, Llist, [[1, 32, 64], [64, 32, 128]], args.device)
else:
    model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

# load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

# init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
# learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

# config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
args.log_dir = log_dir


# res_model ----------------------------------------------------------------------------------------
if args.constraints == 'real':
    from model.func_reg import reg_nonlinear as reg_model_
    reg_model = reg_model_(args.device)
else:
    reg_model = predict(ind=ind, device=args.device)
    reg_model.to(args.device)
    reg_model.load_state_dict(torch.load(args.constraints_net))
    reg_model.train()
# train_res_model(model=reg_model, device=args.device, epoch_num=100)

# start training
trainer = Trainer(model, reg_model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, reg_model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError

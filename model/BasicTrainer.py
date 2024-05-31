import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import numpy as np


def proj(pred_, reg_model, args, step_len=1.0):
    if args.constraints == 'learn':
        # reg_model.train()
        pred = pred_.requires_grad_()
        _ = reg_model(pred)
        reg_model.zero_grad()
        pred.retain_grad()
        reg_model.loss.backward()
        if pred.grad is not None:
            dp = reg_model.loss_item * (-pred.grad) / ((1e-6 + torch.norm(pred.grad, dim=1, keepdim=True)) ** 2) * step_len
            y = pred + dp  # stepsize * dp
        else:
            y = pred
            print('grad is none.')
        return y
    else:
        pred = pred_.requires_grad_()
        rloss = reg_model(pred, sum_=True)
        pred.retain_grad()
        rloss.backward()
        reg = reg_model(pred)
        if pred.grad is not None:
            dp = torch.abs(reg) * (-pred.grad) / ((1e-6 + torch.norm(pred.grad, dim=1, keepdim=True)) ** 2) * step_len
            normdp = torch.norm(dp, dim=1, keepdim=True)
            dp = dp / (1e-6 + normdp)
            stepsize = normdp
            y = pred + stepsize * dp
        else:
            y = pred
            print('grad is none.')
        return y


class Trainer(object):
    def __init__(self, model, reg_model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.reg_model = reg_model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # if not args.debug:
        # self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        # with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            if self.args.iter == 'non':
                with torch.no_grad():
                    output = self.model(data)
                # print(data.size(), label.size(), output.size())
                [a, b, c, d] = list(output.size())
                for p_iter in range(self.args.proj_times):
                    output = torch.reshape(proj(torch.reshape(output, (-1, c)), self.reg_model, self.args, step_len=self.args.step_len), (a, b, c, d)).detach()
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
            else:
                output = []
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                else:
                    label = self.scaler.inverse_transform(label)
                for j in range(self.args.horizon):
                    with torch.no_grad():
                        output_item = self.model(data)
                    # print(data.size(), label.size(), output.size())
                    [a, b, c, d] = list(output_item.size())
                    if not self.args.real_value:
                        output_item = self.scaler.inverse_transform(output_item)
                    for p_iter in range(self.args.proj_times):
                        output_item = torch.reshape(proj(torch.reshape(output_item, (-1, c)), self.reg_model, self.args, step_len=self.args.step_len), (a, b, c, d)).detach()
                    data = torch.cat([data, self.scaler.transform(output_item)], 1)[:, 1:]
                    output.append(np.copy(output_item.cpu().numpy()))
                    # print(output_item.detach().cpu().numpy())
                output = np.concatenate(output, axis=1)
                output = torch.from_numpy(output)
            loss = self.loss(output.cuda(), label)
            # a whole batch of Metr_LA is filtered
            if not torch.isnan(loss):
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        if self.args.constraints == 'learn':
            # frf train 的时候 freeze reg_model
            for param in self.reg_model.parameters():
                param.requires_grad = False
        self.model.train()
        total_loss = 0
        total_mape = 0
        total_mae = 0
        tl = 0
        tlr = 0
        tlrs = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            label = label[:, 0:1]
            [a, b, c, d] = list(data.size())
            # data2 = data * torch.from_numpy(1.0 + self.args.input_noise / 2.0 - self.args.input_noise * np.random.rand(a, 1, c, d).astype(np.float32)).to(self.args.device)
            # for i in range(self.args.noise_input_proj_times):
            #     data2 = torch.reshape(proj(torch.reshape(data2, (-1, c)), self.reg_model, step_len=self.args.step_len), (a, b, c, d)).detach()

            # self.optimizer.zero_grad()

            # teacher_forcing for RNN encoder-decoder model
            # if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            # out_ul = self.model(data2)
            # pred = self.reg_model(torch.reshape(out_ul, (-1, c)))
            # res_loss = torch.mean(torch.abs(pred - self.reg_model.gt) ** 1)
            # res_loss = self.reg_model(torch.reshape(out_ul, (-1, c)), avg_=True)
            # res_loss_ = self.args.resloss_lambda * res_loss
            # self.optimizer.zero_grad()
            # res_loss_.backward()
            # data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss_supervised = self.loss(output.cuda(), label)  # * 0.9 + self.loss(output_origin.cuda(), label) * 0.1
            # reg_loss = 0.0
            # for Ftilde in self.model.reg_state:
            #     reg_loss += self.reg_model(Ftilde, l2_=True)
            pred = self.reg_model(torch.reshape(output, (-1, self.args.num_nodes)))
            reg_loss = torch.mean(torch.abs(pred - self.reg_model.gt) ** 1)
            reg_mape = torch.mean(torch.abs(pred - self.reg_model.gt) / torch.abs(self.reg_model.gt + 1e-6))
            reg_mae = torch.mean(torch.abs(pred - self.reg_model.gt))
            if self.args.reg_lambda > 1e-8:
                loss = loss_supervised + reg_loss * self.args.reg_lambda
            else:
                loss = loss_supervised
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            tl += loss_supervised.item()
            total_mape += reg_mape.item()
            total_mape += reg_mae.item()
            # tlr += reg_loss.item()
            # tlrs += res_loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss_supervised.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        tl_loss = tl / self.train_per_epoch
        total_mape = total_mape / self.train_per_epoch
        total_mae = total_mae / self.train_per_epoch
        # tlr_loss = tlr/self.train_per_epoch
        # tlrs_loss = tlrs/self.train_per_epoch
        # self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(
        #     epoch, train_epoch_loss, teacher_forcing_ratio))
        self.logger.info('**********Train Epoch {}: Loss: {:.6f}, mape: {:.6f}, mape: {:.6f}'.format(
                epoch, tl_loss, total_mae, total_mape))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        np.random.seed(1)
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader is None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.reg_model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, reg_model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        # with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data[..., :args.input_dim]
            label = target[..., :args.output_dim]
            if args.iter == 'non':
                with torch.no_grad():
                    output = model(data)
                [a, b, c, d] = list(output.size())
                for p_iter in range(args.proj_times):
                    output = torch.reshape(proj(torch.reshape(output, (-1, c)), reg_model, args, step_len=args.step_len), (a, b, c, d)).detach()
            else:
                output = []
                for j in range(args.horizon):
                    with torch.no_grad():
                        output_item = model(data)
                    if not args.real_value:
                        output_item = scaler.inverse_transform(output_item)
                    [a, b, c, d] = list(output_item.size())
                    for p_iter in range(args.proj_times):
                        output_item = torch.reshape(proj(torch.reshape(output_item, (-1, c)), reg_model, args, step_len=args.step_len), (a, b, c, d)).detach()
                    output.append(np.copy(output_item.cpu().numpy()))
                    data = torch.cat([data, scaler.transform(output_item)], 1)[:, 1:]
                output = torch.from_numpy(np.concatenate(output, axis=1))
            y_true.append(label)
            y_pred.append(output.detach())
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

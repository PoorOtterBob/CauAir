import os
import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics, masked_f1_score
import sys

class BaseEngine():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, 
                 args):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._args = args
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 
        # sys.exit()

    def _speed_calculation(self, time_list):
        return sum(time_list), sum(time_list) / len(time_list)

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_{}to{}_y{}_s{}.pt'.format(self._args.seq_len, 
                                                      self._args.horizon, 
                                                      self._args.years,
                                                      self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_{}to{}_y{}_s{}.pt'.format(self._args.seq_len, 
                                                      self._args.horizon, 
                                                      self._args.years,
                                                      self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename), map_location=torch.device(self._device)))   

    def present_time(self, train_time_list, val_time_list):
        train_time, train_speed = self._speed_calculation(train_time_list)
        val_time, val_speed = self._speed_calculation(val_time_list)
        self._logger.info('Training Speed: {:.2f}s/epoch'.format(train_speed))               
        self._logger.info('Validating Speed: {:.2f}s/epoch'.format(val_speed))               
        self._logger.info('Time Cost: {:.2f}s'.format(train_time + val_time))    

    def train_batch(self):
        self.model.train()
        cur_lr = None
        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            
            
            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            if self._args.lr_update_in_step:
                if self._lr_scheduler is None:
                    cur_lr = self._lrate
                else:
                    cur_lr = self._lr_scheduler.get_last_lr()[0]
                    self._lr_scheduler.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), cur_lr


    def train(self):
        self._logger.info('Start training!')
        train_time_list, val_time_list = [], []
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, cur_lr = self.train_batch()
            t2 = time.time()
            train_time_list.append(t2 - t1)

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()
            val_time_list.append(v2 - v1)

            if not self._args.lr_update_in_step:
                if self._lr_scheduler is None:
                    cur_lr = self._lrate
                else:
                    cur_lr = self._lr_scheduler.get_last_lr()[0]
                    self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break
                
        self.present_time(train_time_list, val_time_list)
        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))

            f1_score = masked_f1_score(preds, labels)
            log = 'F1 Score for level 0: {:.4f}, 1: {:.4f}, 2: {:.4f}'
            self._logger.info(log.format(f1_score[0], f1_score[1], f1_score[2]))
import torch
import torch.nn as nn
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
from tqdm import tqdm

class Airformer_Engine(BaseEngine):
    def __init__(self, **args):
        super(Airformer_Engine, self).__init__(**args)
        self.rec_mae = nn.L1Loss()
        self.alpha = 1
        
    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X[..., :self._args.input_dim], label]))

            pred, X_rec, kl_loss = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

            rec_loss = self.rec_mae(X_rec[..., :6], X[..., :6])

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss_dec = loss + self.alpha * (rec_loss + kl_loss)
            loss_dec.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            self._iter_cnt += 1
            
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), None
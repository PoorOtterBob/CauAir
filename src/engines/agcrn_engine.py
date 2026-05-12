import torch
import torch.nn as nn
from src.base.engine import BaseEngine
from torch.nn.parallel import DataParallel

class AGCRN_Engine(BaseEngine):
    def __init__(self, **args):
        super(AGCRN_Engine, self).__init__(**args)
        # device_ids = list(range(torch.cuda.device_count()))  
        # self.model = DataParallel(self.model, device_ids=device_ids).to(device_ids[0])  
        # self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3,4,5,6,7])

        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
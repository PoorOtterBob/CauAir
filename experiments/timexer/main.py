import os
import argparse
import numpy as np
import random

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
import torch.nn as nn
torch.set_num_threads(3)
from src.models.timexer import TimeXer 
from src.engines.deepair_engine import DeepAirEngine
from src.utils.args import get_public_config
from src.utils.logging import get_logger
from src.utils.dataloader_deepair import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config():
    parser = get_public_config()
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--features', type=str, default='M')

    # for optimization
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lrate', type=float, default=0.0001)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    
    parser.add_argument('--clip_grad_value', type=float, default=5)

    args = parser.parse_args()

    folder_name = '{}'.format(args.dataset)
    log_dir = './experiments/{}/{}/'.format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def main():
    args, log_dir, logger = get_config()
    device = torch.device(args.device)
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = TimeXer(node_num = node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 configs=args,
                 )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = DeepAirEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed,
                        args=args
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
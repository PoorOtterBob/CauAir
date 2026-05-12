import os
import argparse
import numpy as np
import random

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
import torch.nn as nn
torch.set_num_threads(3)
from src.models.bigst import bigst
from src.engines.bigst_engine import BigST_Engine
from src.utils.args import get_public_config
from src.utils.logging import get_logger
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
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
    parser.add_argument('--adj_type', type=str, default='origin')
    parser.add_argument('--hid_dim', type=int, default=32,help='')
    parser.add_argument('--tau', type=int,default=0.25, help='temperature coefficient')
    parser.add_argument('--random_feature_dim', type=int, default=64, help='random feature dimension')
    parser.add_argument('--node_emb_dim', type=int, default=32, help='node embedding dimension')
    parser.add_argument('--time_emb_dim', type=int, default=32, help='time embedding dimension')
    parser.add_argument('--use_residual', type=bool, default=True, help='use residual connection')
    parser.add_argument('--use_bn', type=bool, default=True, help='use batch normalization')
    parser.add_argument('--use_spatial', type=bool, default=False, help='use spatial loss')
    parser.add_argument('--use_long', type=bool, default=False, help='use long-term preprocessed features')

    # for optimization
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    args = parser.parse_args()

    folder_name = '{}-{}'.format(args.dataset, args.adj_type)
    log_dir = './experiments/{}/{}/'.format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def main():
    args, log_dir, logger = get_config()
    device = torch.device(args.device)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = bigst(node_num = node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 seq_num=args.seq_len,
                 in_dim=args.input_dim, 
                 hid_dim=args.hid_dim, 
                 tau=args.tau, 
                 random_feature_dim=args.random_feature_dim, 
                 node_emb_dim=args.node_emb_dim, 
                 time_emb_dim=args.time_emb_dim, 
                 use_residual=args.use_residual, 
                 use_bn=args.use_bn, 
                 use_spatial=args.use_spatial, 
                 use_long=args.use_long, 
                 dropout=args.dropout, 
                 tod=args.tod
                 )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = BigST_Engine(device=device,
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
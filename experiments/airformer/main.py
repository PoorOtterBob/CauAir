import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.airformer import AirFormer
from src.engines.airformer_engine import Airformer_Engine
from src.utils.args import get_public_config, str_to_bool
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dartboard', type=int, default=0,
                        help='0: 50-200, 1: 50-200-500, 2: 50, 3: 25-100-250')
    parser.add_argument('--stochastic_flag', type=str_to_bool,
                        default=True, help='whether to use stochastic temporal transformer')
    parser.add_argument('--spatial_flag', type=str_to_bool,
                        default=False, help='whether to use spatial transformer')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)

    parser.add_argument('--lrate', type=float, default=5e-4)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)
    
    model = AirFormer(node_num=node_num,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim,
                      seq_len=args.seq_len,
                      horizon=args.horizon,
                      dropout=args.dropout,
                      spatial_flag=args.spatial_flag,
                      stochastic_flag=args.stochastic_flag,
                      hidden_channels=args.n_hidden,
                      dartboard=args.dartboard,
                      end_channels=args.n_hidden * 8,
                      num_heads=args.num_heads,
                      device=device,)

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 9], gamma=args.lr_decay_ratio)

    engine = Airformer_Engine(device=device,
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
                            args=args,
                            )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.timemixer import TimeMixer
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
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
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_layers', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--channel_independence', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--moving_avg', type=int, default=7)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--sigma', type=int, default=16)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--use_norm', type=int, default=0)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--embed', type=str, default='timeF', choices=['timeF', 'fixed', 'learned'], help='Type of embedding')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--time_of_day_size', type=int, default=24, help='Size of time of day feature')
    parser.add_argument('--day_of_week_size', type=int, default=7, help='Size of day of week feature')
    parser.add_argument('--day_of_month_size', type=int, default=31, help='Size of day of month feature')
    parser.add_argument('--day_of_year_size', type=int, default=366, help='Size of day of year feature')


    parser.add_argument('--lrate', type=float, default=0.01)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
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

    args.enc_in = args.dec_in = args.c_out = node_num
    args.label_len = args.seq_len/2
    args.pred_len = args.horizon
    args.num_time_features = args.input_dim
    args.time_of_day_size = args.tod
    model = TimeMixer(node_num=node_num,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim,
                      seq_len=args.seq_len,
                      horizon=args.horizon,
                      model_args=vars(args),
                        )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 25, 50], gamma=0.5)

    engine = BaseEngine(device=device,
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
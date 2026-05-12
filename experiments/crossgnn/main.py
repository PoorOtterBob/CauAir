import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.crossgnn import CrossGNN
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from src.utils.project import normalize_run_dir

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--individual', type=bool, default=False)
    parser.add_argument('--tvechidden', type=int, default=1)
    parser.add_argument('--nvechidden', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=8)
    

    parser.add_argument('--use_ngcn', type=int, default=1)
    parser.add_argument('--use_tgcn', type=int, default=1)
    parser.add_argument('--scale_number', type=int, default=4)
    parser.add_argument('--tk', type=int, default=10)
    parser.add_argument('--anti_ood', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=2)

    parser.add_argument('--lrate', type=float, default=0.0002)
    parser.add_argument('--wdecay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def cont_learning(model, save_path, args):
    save_path = normalize_run_dir(save_path)
    filename = 'final_model_{}to{}_y{}.pt'.format(args.seq_len, 
                                                      args.horizon, 
                                                      args.years,)
    model.load_state_dict(torch.load(
        os.path.join(save_path, filename), map_location=args.device))
    return model

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    # args.time_of_day_size = args.tod = 24 if args.dataset == 'hour' else 1
    args.pred_len = args.horizon
    args.enc_in = args.c_out = node_num

    model = CrossGNN(node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 seq_len=args.seq_len,
                 horizon=args.horizon,
                 configs=args,
                 )
    if args.ct:
        try:
            model = cont_learning(model, log_dir, args)
        except:
            print('No pretrained model!')

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

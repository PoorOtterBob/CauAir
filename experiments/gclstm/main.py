import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.gclstm import GCLSTM
from src.engines.deepair_engine import DeepAirEngine
from src.utils.args import get_public_config
from src.utils.dataloader_deepair import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
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

    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/{}'.format(args.model_name, args.dataset, args.years)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def load_and_fixed_well_trained_stmodel(model, args, log_dir):
    log_dir = normalize_run_dir(log_dir)
    filename = filename = 'final_model_{}to{}_s{}.pt'.format(args.seq_len, 
                                                             args.horizon, 
                                                             args.seed)
    model.load_state_dict(torch.load(os.path.join(log_dir, filename)))
    print('load model done')
    return model


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = GCLSTM(node_num=node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                seq_len=args.seq_len,
                horizon=args.horizon,
                gso=gso,
                )
    
    if args.ct:
        model = load_and_fixed_well_trained_stmodel(model, args, log_dir)
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 25, 50], gamma=0.5)

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
                        args=args,
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()

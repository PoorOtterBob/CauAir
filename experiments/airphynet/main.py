import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.airphynet import AirPhyNetModel
from src.base.engine import BaseEngine
from src.utils.args import get_public_config, str_to_bool
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from tqdm import tqdm

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
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--lrate', type=float, default=0.0005)
    parser.add_argument('--wdecay', type=float, default=0.0005)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def edge_index_edge_attr(adj_matrix):
    e = np.count_nonzero(adj_matrix)

    edge_index = np.zeros((e, 2), dtype=int)
    edge_attr = np.zeros((e, 2))

    count = 0
    for i in tqdm(range(len(adj_matrix))):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                edge_index[count][0] = i
                edge_index[count][1] = j
                edge_attr[count][0] = adj_matrix[i][j]
                edge_attr[count][1] = adj_matrix[i][j]
                # edge_attr[count] = adj_matrix[i][j]
                count += 1
    print('Edge number ', count)
    return edge_index, edge_attr

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    np.fill_diagonal(adj_mx, 1)
    edge_index, edge_attr = edge_index_edge_attr(adj_mx)    
    dataloader, scaler = load_dataset(data_path, args, logger)
    
    model = AirPhyNetModel(node_num=node_num,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim,
                      seq_len=args.seq_len,
                      horizon=args.horizon,
                      adj_mx=adj_mx,
                      edge_index=edge_index,
                      edge_attr=edge_attr,
                      logger=logger,
                      device=device,)

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40, 50], gamma=0.1)

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
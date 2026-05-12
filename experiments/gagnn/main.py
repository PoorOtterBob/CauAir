import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.gagnn import GAGNN
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info, load_adj_from_numpy
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from src.utils.project import DATA_ROOT, KNOWAIR_ROOT, GAGNN_ROOT
from sklearn.cluster import KMeans
from torch_geometric.nn import MetaLayer
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--modde', type=str, default='full', help='')
    parser.add_argument('--encoder', type=str, default='self', help='')
    parser.add_argument('--w_init', type=str, default='rand', help='')
    parser.add_argument('--mark', type=str, default='',help='')
    parser.add_argument('--w_rate', type=int, default=50, help='')
    parser.add_argument('--city_num', type=int, default=209, help='')
    parser.add_argument('--group_num', type=int, default=15, help='')
    parser.add_argument('--gnn_h', type=int, default=32, help='')
    parser.add_argument('--gnn_layer', type=int, default=2, help='')
    parser.add_argument('--x_em', type=int, default=32, help='x embedding')
    parser.add_argument('--date_em', type=int, default=4, help='date embedding')
    parser.add_argument('--loc_em', type=int, default=12, help='loc embedding')
    parser.add_argument('--edge_h', type=int, default=12, help='edge h')


    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger

def edge_index_edge_attr(adj_matrix):
    e = np.count_nonzero(adj_matrix)

    edge_index = np.zeros((2, e), dtype=int)
    edge_attr = np.zeros((e, 1))

    count = 0
    for i in tqdm(range(len(adj_matrix))):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                edge_index[0][count] = i
                edge_index[1][count] = j
                edge_attr[count] = adj_matrix[i][j]
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
    # np.fill_diagonal(adj_mx, 1)
    edge_index, edge_attr = edge_index_edge_attr(adj_mx)
    dataloader, scaler = load_dataset(data_path, args, logger)

    w = None
    args.city_num = node_num
    if node_num == 1341:
        city_loc = np.load(str(DATA_ROOT / 'loc.npy'))
    elif node_num == 184:
        city_loc = np.load(str(KNOWAIR_ROOT / 'loc.npy'))
    elif node_num == 209:
        city_loc = np.load(str(GAGNN_ROOT / 'loc.npy')).astype(float)
    if args.w_init == 'group':
        # city_loc = np.load(os.path.join('loc.npy'), allow_pickle=True)
        kmeans = KMeans(n_clusters=args.group_num, random_state=0).fit(city_loc)
        group_list = kmeans.labels_.tolist()
        w = np.random.randn(args.city_num, args.group_num)
        w = w * 0.1
        for i in range(len(group_list)):
            w[i,group_list[i]] = 1.0
        w = torch.FloatTensor(w).to(device, non_blocking=True)
    model = GAGNN(node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 seq_len=args.seq_len,
                 horizon=args.horizon,
                 mode=args.modde,
                 encoder=args.encoder,
                 w_init=args.w_init,
                 w=w,
                 x_em=args.x_em,
                 date_em=args.date_em,
                 loc_em=args.loc_em,
                 edge_h=args.edge_h,
                 gnn_h=args.gnn_h,
			     gnn_layer=args.gnn_layer,
                 city_num=args.city_num,
                 group_num=args.group_num,
                 pred_step=args.horizon,
                 device=device,
                 edge_index=edge_index,
			     edge_w=edge_attr,
			     loc=city_loc,
                 )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

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

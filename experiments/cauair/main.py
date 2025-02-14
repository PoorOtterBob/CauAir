import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.cauair import CauAir
from src.engines.cauair_engine import CauAir_Engine
from src.utils.args import get_cauair_config
from src.utils.dataloader_cauair import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def cont_learning(model, save_path, args):
    filename = 'final_model_{}to{}_y{}_s{}_rank{}_head{}_dim{}.pt'.format(args.seq_len, 
                                                      args.horizon, 
                                                      args.years,
                                                      args.seed,
                                                      args.rank,
                                                      args.head,
                                                      args.dim)
    model.load_state_dict(torch.load(
        os.path.join(save_path, filename), map_location=args.device))
    return model

def get_config():
    parser = get_cauair_config()
    parser.add_argument('--lrate', type=float, default=1e-2)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}_rank{}_head{}_dim{}.log'.format(args.seed,
                                                                                       args.rank,
                                                                                        args.head,
                                                                                        args.dim))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    args.seed = torch.randint(999999, (1,)).item() # set random seed here
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = CauAir(node_num=node_num,
                   input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   seq_len=args.seq_len,
                   horizon=args.horizon,
                   dim=args.dim,
                   rank=args.rank,
                   head=args.head,
                   )
    
    if args.ct:
        try:
            model = cont_learning(model, log_dir, args)
        except:
            print('No pretrained model!')

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5)
    args.lr_update_in_step = 0
    engine = CauAir_Engine(device=device,
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
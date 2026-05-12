from __future__ import annotations

import argparse

def get_cauair_config():
    parser = get_public_config()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    return parser


def get_public_config():
    """Common arguments shared by the main forecasting experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='24_24')
    parser.add_argument('--years', type=str, default='all')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2025)

    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--tod', type=int, default=24)

    parser.add_argument('--ct', type=int, default=0)
    parser.add_argument('--lr_update_in_step', type=int, default=0)
    return parser


def get_yuki_config():
    """Public config for the Yuki / XBLZZB family of experiments."""
    parser = get_public_config()
    parser.add_argument('--w_rate', type=int, default=50)
    parser.add_argument('--city_num', type=int, default=209)
    parser.add_argument('--group_num', type=int, default=15)
    parser.add_argument('--gnn_h', type=int, default=32)
    parser.add_argument('--gnn_layer', type=int, default=2)
    parser.add_argument('--x_em', type=int, default=32)
    parser.add_argument('--date_em', type=int, default=4)
    parser.add_argument('--loc_em', type=int, default=12)
    parser.add_argument('--edge_h', type=int, default=12)
    parser.add_argument('--modde', type=str, default='full')
    parser.add_argument('--encoder', type=str, default='self')
    parser.add_argument('--w_init', type=str, default='rand')
    parser.add_argument('--mark', type=str, default='')
    return parser


def get_xyq_config():
    """Public config for the STRIP/BIST family of experiments."""
    parser = get_public_config()
    parser.add_argument('--core', type=int, default=8)
    parser.add_argument('--same', type=int, default=0)
    parser.add_argument('--shap', type=int, default=0)
    parser.add_argument('--cl_epoch', type=int, default=3)
    parser.add_argument('--warm_epoch', type=int, default=30)
    return parser


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

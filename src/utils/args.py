import argparse

def get_cauair_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='24_24')
    parser.add_argument('--years', type=str, default='all')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default='2025')

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--tod', type=int, default=24)

    parser.add_argument('--ct', type=int, default=0) # continue learning

    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    return parser


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
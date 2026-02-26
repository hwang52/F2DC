from argparse import ArgumentParser
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='dataset for experiments')

    parser.add_argument('--model', type=str, required=True, default='f2dc', 
                        help='method name', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')

    parser.add_argument('--optim_wd', type=float, default=0., 
                        help='weight decay')

    parser.add_argument('--optim_mom', type=float, default=0., 
                        help='momentum')

    parser.add_argument('--optim_nesterov', type=int, default=0, 
                        help='nesterov momentum')    

    parser.add_argument('--n_epochs', type=int, help='epochs')

    parser.add_argument('--batch_size', type=int, help='batch size')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true', 
                        help='csv logging', default=False)
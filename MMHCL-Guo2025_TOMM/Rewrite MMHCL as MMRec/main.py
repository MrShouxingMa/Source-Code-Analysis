# coding: utf-8
import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MMHCL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='sports', help='Choose a dataset from {tiktok,sports,clothing}')
    parser.add_argument('--gpu_id', '-g', type=int, default=1, help='gpu number')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

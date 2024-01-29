import argparse
import os
import sys
import torch
import random
import numpy as np
from exp.exp_main import Exp_Main


def main():

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--model', type=str, default='HeterogeneousModel',
                        help='Model type, options: [HeterogeneousModel, CNNModel, GRUModel, LSTMModel]')
    parser.add_argument('--LOSSFACTOR', type=float, default=0.0, help='Weight of trend difference loss')
    parser.add_argument('--QUANTILE', type=float, default=0.5, help='Quantile value of the quantile loss')
    parser.add_argument('--dataset', type=str, default='track3_train.pkl', help='Name of the dataset')
    
    
    # path config
    parser.add_argument('--DATA_PATH', type=str, default='dataset', help='Path of the dataset')
    parser.add_argument('--MODEL_SAVE_PATH', type=str, default='checkpoints/basemodel', help='Model save path')
    parser.add_argument('--RESULT_SAVE_PATH', type=str, default='test_results/basemodel', help='Result save path')
    
    # model define
    parser.add_argument('--num_epochs', type=int, default=10000, help='epoch')
    parser.add_argument('--early_stop', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    
    parser.add_argument('--feature_size', type=int, default=86, help='Number of features')
    parser.add_argument('--hidden_size', type=int, default=512, help='The number of hidden layers in GRU')
    parser.add_argument('--output_size', type=int, default=4, help='Number of output')
    parser.add_argument('--num_layers', type=int, default=1, help='The number of layers in GRU')
    parser.add_argument('--timestep', type=int, default=48)
    
    parser.add_argument('--nfold', type=int, default=5)
    
    
    # target config
    parser.add_argument('--TARGET_FEATS', type=str, default='wdir_2min,spd_2min,spd_inst_max', 
                        help='an array of targets')
    
    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    # setting record of experiments
    setting = '{}_lf{}_q{}_lr{}_fs{}_os{}'.format(
        args.model,
        args.LOSSFACTOR,
        args.QUANTILE,
        args.learning_rate,
        args.feature_size,
        args.output_size)
    
    Exp = Exp_Main

    exp = Exp(args)  # set experiments
    exp.run()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

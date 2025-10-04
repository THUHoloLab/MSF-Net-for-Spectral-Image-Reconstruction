import argparse
import os
from os import mkdir


def parse_args():
    file_dir=os.path.dirname(os.path.realpath(__file__))
    parent_dir=os.path.dirname(file_dir)

    parser = argparse.ArgumentParser(description='Parameter settings........')
    parser.add_argument('--train_dir', type=str,
                        default=os.path.join(parent_dir,'train_dataset'),
                        help='Directory of training input images')
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(parent_dir,'test_dataset'))
    parser.add_argument('--net_save_dir', type=str, default=os.path.join(file_dir,'model_save'),
                        help='Directory for saving dict file')
    parser.add_argument('--image_save_dir', type=str, default=os.path.join(file_dir,'image_save'),
                        help='Directory for saving TensorBoard file')
    parser.add_argument('--save_freq', type=int, default=20, help='Interval to save model')
    parser.add_argument('--log_freq', type=int, default=20, help='Interval to write to TensorBoard')
    parser.add_argument('--spectral_channel', type=int, default=29, help='Number of spectral channels')
    parser.add_argument('--measurement_channel', type=int, default=1, help='Number of spectral channels')
    parser.add_argument('--image_size', type=int, default=256, help='image_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_worker')
    parser.add_argument('--pin_memory', type=str, default='False', help='pin_memory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchSize', type=int, default=5, help='batchSize')
    parser.add_argument('--n_epoch', type=float, default=2500, help="number of epochs of training")
    args = parser.parse_args([])

    print(args)
    return args

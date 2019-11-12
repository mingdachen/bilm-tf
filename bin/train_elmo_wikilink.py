
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import WikiLinkDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 64  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 262,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 600,  # 4096
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 300,  # 512
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 1,  # 10
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 50,
        'bow_size': 50,
        'n_negative_samples_batch': 1024,  # 8192
    }

    prefix = args.train_prefix
    data = WikiLinkDataset(
        vocab=vocab,
        filepattern=prefix,
        path2ent2def=args.desc_path,
        num_steps=options['unroll_steps'],
        bow_size=options['bow_size'])

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, args.restart_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--desc_path', help='path to wiki descriptions')
    parser.add_argument('--restart_checkpoint', help='path to checkpoint')

    args = parser.parse_args()
    main(args)

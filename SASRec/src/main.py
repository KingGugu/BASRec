# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np

from datasets import SASRecDataset
from trainers import SASRecTrainer
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from models import SASRecModel, OfflineItemSimilarity, OnlineItemSimilarity


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            info = f"{arg:<30} : {getattr(args, arg):>35}"
            print(info)
            f.write(info + '\n')


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=1, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # data augmentation args (from CoSeRec)
    parser.add_argument('--noise_ratio', default=0.0, type=float,
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', default=1.0, type=float,
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str,
                        help="Method to generate item similarity score. choices: Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec")
    parser.add_argument('--base_augment_type', default='random', type=str,
                        help="default data augmentation types. Chosen from: reorder, substitute, random.")

    # model args
    parser.add_argument("--model_name", default='M4SRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2024, type=int)

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # No use, we disable this module implemented by CoSeRec because is time-costing
    parser.add_argument("--augmentation_warm_up_epoches", type=float, default=500,
                        help="number of epochs to switch from memory-based similarity model to hybrid similarity model.")

    ############################################
    # need adjust (epoch)
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--pretrain_epoch", default=0, type=int)
    parser.add_argument("--start_valid", default=100, type=int)
    parser.add_argument("--patience", type=int, default=20, help="early stop patience")

    # need adjust (parameter)
    parser.add_argument('--n_pairs', default=1, type=int, metavar='N',
                        help='Number of augmented data for each sequence, one pair equals two augment sequences.')
    parser.add_argument('--n_whole_level', default=1, type=int, metavar='N',
                        help='Number of whole level mix learning.')
    parser.add_argument("--wml_weight", type=float, default=1.0, help="weight of whole level mix learning task")
    parser.add_argument("--aml_weight", type=float, default=1.0, help="weight of augment sequence mix learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of main recommendation task")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="hidden dropout p")
    parser.add_argument("--beta", type=float, default=0.3, help="beta value distributions for beta")
    parser.add_argument("--rate_min", type=float, default=0.2, help="min ratio for two operators")
    parser.add_argument("--rate_max", type=float, default=0.51, help="max ratio for two operators")
    ############################################

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 2

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    show_args_info(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,
                                              args.data_name + '_' + args.similarity_model_name + '_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                                                     similarity_path=args.similarity_model_path,
                                                     model_name=args.similarity_model_name,
                                                     dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model

    # -----------   online based on shared item embedding for item similarity --------- #
    online_similarity_model = OnlineItemSimilarity(item_size=args.item_size)
    args.online_similarity_model = online_similarity_model

    # training data for node classification
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = SASRecModel(args=args)

    if args.load_pretrain:
        model.load_state_dict(torch.load(args.data_dir + 'M4SRec-' + args.data_name + '-2.pt'))

    trainer = SASRecTrainer(model, train_dataloader, eval_dataloader,
                            test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        if args.load_pretrain:
            print('Performance of pre-trained model on the validation set:')
            trainer.valid(0, full_sort=True)
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            if epoch > args.start_valid:
                scores, _ = trainer.valid(epoch, full_sort=True)
                early_stopping(np.array(scores[-1:]), trainer.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        trainer.args.train_matrix = test_rating_matrix
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()

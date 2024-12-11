# -*- coding: utf-8 -*-

import torch

from utils import neg_sample
from torch.utils.data import Dataset
from data_augmentation import Reorder, Substitute, Random


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, data_type='train', similarity_model_type='offline'):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        if similarity_model_type == 'offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type == 'online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type == 'hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'reorder': Reorder(args.rate_min, args.rate_max),
                              'substitute': Substitute(self.similarity_model, args.rate_min, args.rate_max),
                              'random': Random(self.similarity_model, args.rate_min, args.rate_max)}
        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        self.n_pairs = self.args.n_pairs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _one_pair_data_augmentation(self, input_ids):
        '''
        provides two positive samples given one sequence
        '''
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids, aug_weight = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len:]
            assert len(augmented_input_ids) == self.max_len
            cur_tensors = (
                torch.tensor(augmented_input_ids, dtype=torch.long),
                torch.tensor(aug_weight, dtype=torch.float)
            )
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use
            rec_batch = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cl_batch = []
            total_augmentaion_pairs = self.n_pairs
            for i in range(total_augmentaion_pairs):
                cl_batch.append(self._one_pair_data_augmentation(input_ids))
            return (rec_batch, cl_batch)

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)

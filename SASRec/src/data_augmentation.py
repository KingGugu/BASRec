# -*- coding: utf-8 -*-

import copy
import random
import numpy as np


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, item_similarity_model, rate_min, rate_max):
        self.data_augmentation_methods = [Reorder(rate_min, rate_max),
                                          Substitute(item_similarity_model, rate_min, rate_max)]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        return augment_method(sequence)


def _ensmeble_sim_models(top_k_one, top_k_two):
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Substitute(object):
    """Substitute with similar items"""

    def __init__(self, item_similarity_model, rate_min, rate_max):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.rate_min = rate_min
        self.rate_max = rate_max

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        substitute_rate = np.random.uniform(low=self.rate_min, high=self.rate_max)
        substitute_nums = max(int(substitute_rate * len(copied_sequence)), 1)

        # substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k=substitute_nums)
        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:

                copied_sequence[index] = copied_sequence[index] = \
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence, substitute_rate


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, rate_min, rate_max):
        self.rate_min = rate_min
        self.rate_max = rate_max

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        reorder_rate = np.random.uniform(low=self.rate_min, high=self.rate_max)
        sub_seq_length = int(reorder_rate * len(copied_sequence))

        # sub_seq_length = int(self.reorder_rate * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq, reorder_rate

# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.distributions as dist

from tqdm import tqdm
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        if self.cuda_condition:
            self.model.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.n_whole_level = args.n_whole_level
        self.beta_distribution = dist.Beta(torch.tensor([args.beta]), torch.tensor([args.beta]))
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids, weight=None):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        pos_logits = (seq_out * pos_emb).sum(dim=-1).view(-1)
        neg_logits = (seq_out * neg_emb).sum(dim=-1).view(-1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        if weight is not None:
            weight = weight.unsqueeze(-1)
            pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            pos_loss = pos_loss * weight
            neg_loss = neg_loss * weight
            loss = torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
        else:
            loss = torch.sum(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "valid"
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, aug_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                joint_loss = self.args.rec_weight * rec_loss

                # ---------- augmentation -------------#
                aug_losses = []
                if epoch >= self.args.pretrain_epoch:
                    for aug_batch in aug_batches:
                        aug_loss = self.aug_mix_learning(input_ids, aug_batch, target_pos, target_neg)
                        aug_losses.append(aug_loss)
                        joint_loss += self.args.aml_weight * aug_loss
                    whole_mix_loss = self.whole_mix_learning(sequence_output, target_pos, target_neg, input_ids)
                    joint_loss += self.args.wml_weight * whole_mix_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()
                rec_avg_loss += rec_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                 desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                 total=len(dataloader),
                                 bar_format="{l_bar}{r_bar}")
            self.model.eval()
            pred_list = None
            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)

    def whole_mix_learning(self, seq_out, pos_ids, neg_ids, input_ids):

        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        mix_rec_loss = 0

        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0], seq_out.shape[1])).to(self.device)
            indices = torch.randperm(seq_out.shape[0])

            mixed_seq_out = alpha * seq_out + (1 - alpha) * seq_out[indices]
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        for i in range(self.n_whole_level):
            alpha = self.beta_distribution.sample((seq_out.shape[0], seq_out.shape[2])).permute(0, 2, 1).to(self.device)
            indices = torch.randperm(seq_out.shape[0])

            mixed_seq_out = alpha * seq_out + (1 - alpha) * seq_out[indices]
            mixed_pos_emb = alpha * pos_emb + (1 - alpha) * pos_emb[indices]
            mixed_neg_emb = alpha * neg_emb + (1 - alpha) * neg_emb[indices]
            mix_pos_logits = (mixed_seq_out * mixed_pos_emb).sum(dim=-1).view(-1)
            mix_neg_logits = (mixed_seq_out * mixed_neg_emb).sum(dim=-1).view(-1)

            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
            mix_rec_loss += torch.sum(
                - torch.log(torch.sigmoid(mix_pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(mix_neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)

        return mix_rec_loss

    def aug_mix_learning(self, input_ids, aug_inputs, target_pos, target_neg):
        """
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        aug_loss = 0
        for aug_batch in aug_inputs:
            aug_sequences, aug_weight = aug_batch
            aug_weight = aug_weight.to(self.device)
            aug_sequences = aug_sequences.to(self.device)
            mix_weight = self.beta_distribution.sample(torch.tensor([input_ids.shape[0]])).unsqueeze(-1).to(self.device)

            loss_weight = 1.0 / (mix_weight.squeeze(-1).squeeze(-1) * aug_weight)
            loss_weight = (loss_weight - loss_weight.min()) / (loss_weight.max() - loss_weight.min())

            augment_output = self.model.transformer_encoder(input_ids, aug_sequences, mix_weight)
            aug_loss += self.cross_entropy(augment_output, target_pos, target_neg, loss_weight)
        return aug_loss

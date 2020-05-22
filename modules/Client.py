import numpy as np
import random
from collections import defaultdict, deque


class Client:
    def __init__(self, client_id, model, train, train_user_list, validation_user_list, test_user_list, sampler_size):
        self.id = client_id
        self.model = model
        self.train_set = train
        self.train_user_list = train_user_list
        self.validation_user_list = validation_user_list
        self.test_user_list = test_user_list
        self.sampler_size = sampler_size

    def predict(self, max_k):
        result = self.model.predict()
        result[list(self.train_user_list)] = -np.inf
        top_k = result.argsort()[-max_k:][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train_old(self, lr, positive_fraction, most_popular_items, step):
        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200
        resulting_dic = {}
        resulting_bias = {}

        if most_popular_items:
            self.train_set.set_selection_list(most_popular_items, step)
        for i, j in self.train_set.sample_user_triples():
            x_i = self.model.predict_one(i)
            x_j = self.model.predict_one(j)
            x_ij = x_i - x_j

            d_loss = 1 / (1 + np.exp(x_ij))

            bi = self.model.item_bias[i].copy()
            bj = self.model.item_bias[j].copy()
            bi_new = (d_loss - bias_reg * bi)
            bj_new = (-d_loss - bias_reg * bj)

            self.model.item_bias[i] += lr * bi_new
            self.model.item_bias[j] += lr * bj_new

            wu = self.model.user_vec.copy()
            hi = self.model.item_vecs[i].copy()
            hj = self.model.item_vecs[j].copy()
            hi_new = (d_loss * wu - positive_item_reg * hi)
            hj_new = (d_loss * (-wu) - negative_item_reg * hj)

            self.model.user_vec += lr * (d_loss * (hi - hj) - user_reg * wu)
            self.model.item_vecs[i] += lr * hi_new
            self.model.item_vecs[j] += lr * hj_new

            resulting_dic[j] = hj_new
            resulting_bias[j] = bj_new
            if positive_fraction:
                if random.random() >= 1 - positive_fraction:
                    resulting_dic[i] = hi_new
                    resulting_bias[i] = bi_new

        return resulting_dic, resulting_bias



    def train(self, lr, positive_fraction, most_popular_items, step):
        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200
        resulting_dic = defaultdict(lambda: np.zeros(len(self.model.user_vec)))
        resulting_bias = defaultdict(float)


        if most_popular_items:
            self.train_set.set_selection_list(most_popular_items, step)

        sample = self.train_set.sample_user_triples()
        i, j = sample.__next__()
        #x_i = self.model.predict_one(i)
        x_j = self.model.predict_one(j)
        #x_ij = x_i - x_j
        x_ij = - x_j
        d_loss = 1 / (1 + np.exp(x_ij))

        bj_new = (-d_loss - bias_reg * self.model.item_bias[j])
        wu = self.model.user_vec.copy()

        hj_new = (d_loss * (-wu) - negative_item_reg * self.model.item_vecs[j])
        #self.model.user_vec += lr * (d_loss * (self.model.item_vecs[i] - self.model.item_vecs[j]) - user_reg * wu)
        self.model.user_vec += lr * (d_loss * (- self.model.item_vecs[j]) - user_reg * wu)

        resulting_dic[j] = np.add(resulting_dic[j], hj_new)
        resulting_bias[j] += bj_new

        d_wu = d_loss * (-wu)
        j_samples = [j for _, j in sample]
        deque(map(lambda j: resulting_dic.update({j: np.add(resulting_dic[j], d_wu - negative_item_reg * self.model.item_vecs[j])}),
            j_samples), maxlen=0)
        deque(map(lambda j: resulting_bias.update({j: resulting_bias[j] - d_loss - bias_reg * self.model.item_bias[j]}), j_samples), maxlen=0)
        # for i, j in sample:
        #     bj_new = (-d_loss - bias_reg * self.model.item_bias[j])
        #     hj_new = (d_loss * (-wu) - negative_item_reg * self.model.item_vecs[j])
        #     resulting_dic[j] = np.add(resulting_dic[j], hj_new)
        #     resulting_bias[j] += bj_new

        return resulting_dic, resulting_bias

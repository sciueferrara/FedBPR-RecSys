import numpy as np

np.random.seed(43)


class TripletSampler:
    def __init__(self, train_user_list, item_size, sampler_size):
        self.train_user_list = list(train_user_list)
        self.item_size = item_size
        self.sampler_size = sampler_size
        self.selection_list = None

    def set_selection_list(self, most_popular_items, step):
        if most_popular_items[0] == 1:
            s = int(step * 0.1 * len(most_popular_items[1]))
            e = int((step + 1) * 0.1 * len(most_popular_items[1]))
        elif most_popular_items[0] == 2:
            s = 0
            e = int((step + 1) * 0.1 * len(most_popular_items[1]))
        self.selection_list = list(set(self.train_user_list) & set(most_popular_items[1][s:e]))

    def sample_user_triples(self):
        for _ in range(self.sampler_size):
            if self.selection_list:
                i = np.random.choice(self.selection_list)
            else:
                i = np.random.choice(self.train_user_list)
            j = np.random.randint(self.item_size)
            while j in self.train_user_list:
                j = np.random.randint(self.item_size)
            yield i, j

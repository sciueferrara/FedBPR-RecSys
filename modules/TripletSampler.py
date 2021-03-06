import numpy as np

np.random.seed(43)


class TripletSampler:
    def __init__(self, train_user_list, item_size, sampler_size):
        self.train_user_list = list(train_user_list)
        self.item_size = item_size
        self.sampler_size = sampler_size
        self.selection_list = None
        self.set_of_selection_list = None
        self.set_of_train_list = set(train_user_list)

    def set_selection_list(self, most_popular_items, step):
        if most_popular_items[0] == 1:
            s = int(step * 0.1 * len(most_popular_items[1]))
            e = int((step + 1) * 0.1 * len(most_popular_items[1]))
        elif most_popular_items[0] == 2:
            s = 0
            e = int((step + 1) * 0.1 * len(most_popular_items[1]))
        elif most_popular_items[0] == 3:
            s = 0
            if step == 0:
                e = int(0.5 * len(most_popular_items[1]))
            elif step == 1:
                e = int(0.7 * len(most_popular_items[1]))
            elif step == 2:
                e = int(0.9 * len(most_popular_items[1]))
            elif step == 3:
                e = len(most_popular_items[1])
        # self.selection_list = list(set(self.train_user_list) & set(most_popular_items[1][s:e]))
        self.selection_list = most_popular_items[1][s:e]
        self.set_of_selection_list = set(most_popular_items[1][s:e])

    def sample_user_triples(self):
        #i = np.random.choice(self.train_user_list)
        for _ in range(self.sampler_size):
            j = np.random.randint(self.item_size)
            if self.selection_list:
                while j not in self.set_of_selection_list or j in self.set_of_train_list:
                    j = np.random.randint(self.item_size)
            else:
                while j in self.set_of_train_list:
                    j = np.random.randint(self.item_size)
            yield j

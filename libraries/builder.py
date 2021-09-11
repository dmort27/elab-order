import copy

from phonepiece.cart.tree import TreeClassifier
from phonepiece.cart.node import Node, Tree
from phonepiece.inventory import read_inventory
import tqdm
from phonepiece.unit import read_unit
from collections import Counter
import numpy as np

def train_lang(lang_id):

    config = {
        'features': [1,3],
        'impurity': 0.05,
        'depth': 25,
    }

    builder = TreeBuilder(lang_id, config)

    char_unit = read_unit('../data/cart/'+lang_id+'/char.txt')
    phone_unit = read_inventory(lang_id).phoneme

    X = []
    y = []

    for line in open('../data/cart/' +lang_id+'/train.txt', 'r'):
        fields = line.strip().split()
        phone = fields[0]
        y.append(phone_unit.get_joint_id(phone))
        X.append(char_unit.get_joint_ids(fields[1:]))

    X = np.array(X)
    y = np.array(y)

    tree = builder.fit(X, y)
    print(len(tree))

    return tree

def test_lang(lang_id, model):

    char_unit = read_unit('../data/cart/'+lang_id+'/char.txt')
    phone_unit = read_inventory(lang_id).phoneme

    all_cnt = 0
    correct_cnt = 0

    for line in tqdm.tqdm(open('../data/cart/'+lang_id+'/test.txt', 'r').readlines()):

        fields = line.strip().split()
        phone = fields[0]
        y = phone_unit.get_joint_id(phone)
        x = char_unit.get_joint_ids(fields[1:])
        ans = model.classify(x)

        if y == ans:
            correct_cnt += 1
        #else:
        #    print("expect ", y, " but ", ans)
        all_cnt += 1

    if all_cnt == 0:
        return -1
    return correct_cnt / all_cnt


class TreeBuilder:

    def __init__(self, lang_id, config):
        self.lang_id = lang_id
        self.config = config

        if 'features' in config:
            self.features = config['features']
        else:
            self.features = [1,3]

        if 'depth' in config:
            self.depth = config['depth']
        else:
            self.depth = 25

        if 'impurity' in config:
            self.impurity = config['impurity']
        else:
            self.impurity = 0.1

        self.feature_size = -1
        self.sample_size = -1

    def fit(self, X, y):

        meta_dict = {'lang_id': self.lang_id,
                     'sample_size': len(X),
                     'score': 0.0,
                     }

        self.root = Tree({}, meta_dict)

        x = X[:,2]

        center_ids = np.unique(x)

        for center_id in center_ids:
            mask = x == center_id
            sub_x = X[mask,:]
            sub_y = y[mask]

            node = Node(None)
            self.split(node, sub_x, sub_y, 0)
            self.root.add_top_node(center_id, node)

        return self.root

    def impurity_score(self, y, idx_mask=None):

        if idx_mask is None:
            p = np.array(list(Counter(y).values())) / len(y)
            return np.sum(p*(1-p))

        y1 = y[idx_mask]
        y2 = y[~idx_mask]

        y1_size = len(y1)
        y2_size = len(y2)
        all_size = len(y)

        p1 = np.array(list(Counter(y1).values()))/y1_size
        impurity_score1 = np.sum(p1*(1-p1))*y1_size/all_size

        p2 = np.array(list(Counter(y2).values()))/y2_size
        impurity_score2 = np.sum(p2*(1-p2))*y2_size/all_size

        return impurity_score1 + impurity_score2


    def split(self, node, X, y, depth):

        # stop splitting when impurity is too small
        if len(y) < 5 or self.impurity_score(y) < self.impurity or depth >= self.depth:
            # set the most frequent label
            try:
                node.label = list(Counter(y).keys())[0]
            except:
                node.label = 0

            return

        best_feature_index = -1
        best_impurity_score = 1
        best_feature_cat_set = None
        best_feature_mask = None

        for feature_index in self.features:
            x = X[:,feature_index]

            cats = np.unique(x)
            cat_scores = []

            # compute impurity for each category
            for i, cat in enumerate(cats):
                idx_mask = x == cat

                impurity_score = self.impurity_score(y, idx_mask)
                cat_scores.append((impurity_score, cat))

            cat_scores.sort()
            mask = np.zeros(len(x), dtype=bool)
            cat_set = []

            best_cat_set = []
            best_mask = None
            best_score = 1.0

            for _, cat in cat_scores:

                cat_set.append(cat)

                # update mask
                mask |= x == cat
                impurity_score = self.impurity_score(y, mask)

                if impurity_score < best_score:
                    best_score = impurity_score
                    best_cat_set = copy.deepcopy(cat_set)
                    best_mask = copy.deepcopy(mask)

            if best_score < best_impurity_score:
                best_impurity_score = best_score
                best_feature_index = feature_index
                best_feature_cat_set = best_cat_set
                best_feature_mask = best_mask

        if abs(best_impurity_score - self.impurity_score(y)) < 0.001:
            node.label = list(Counter(y).keys())[0]
            return

        node.left_set = set(list(best_feature_cat_set))
        node.feature_idx = best_feature_index

        left_x = X[best_feature_mask, :]
        right_x = X[~best_feature_mask, :]
        left_y = y[best_feature_mask]
        right_y = y[~best_feature_mask]

        node.left = Node(self.config)
        node.right = Node(self.config)

        self.split(node.left, left_x, left_y, depth+1)
        self.split(node.right, right_x, right_y, depth+1)
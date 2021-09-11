import json
from phonepiece.config import phonepiece_config
from phonepiece.inventory import read_inventory
from phonepiece.grapheme import read_grapheme
from phonepiece.unit import read_unit
import numpy as np

def rec_read_cart_tree(tree_dict):

    node = Node(None)

    if tree_dict['left_set'] is not None:
        node.left_set = set(tree_dict['left_set'])

    if tree_dict['label'] is not None:
        node.label = int(tree_dict['label'])

    if tree_dict['left']:
        node.left = rec_read_cart_tree(tree_dict['left'])

    if tree_dict['right']:
        node.right = rec_read_cart_tree(tree_dict['right'])

    node.feature_idx = int(tree_dict['feature_idx'])

    return node


def read_cart_tree(path):
    tree_dict = json.load(open(path, 'r'))

    meta_dict = tree_dict['meta']

    id2node = {}
    for k, v in tree_dict['id2node'].items():
        id2node[int(k)] = rec_read_cart_tree(v)

    tree = Tree(id2node, meta_dict)

    return tree


def read_cart(lang_id):
    return read_cart_tree(phonepiece_config.data_path / 'cart' / lang_id / 'tree.json')


def build_cart_dict(node):

    node_dict = {}

    if node.left_set is not None:
        node_dict['left_set'] = sorted(map(int, list(node.left_set)))
    else:
        node_dict['left_set'] = []

    if node.label is not None:
        node_dict['label'] = int(node.label)
    else:
        node_dict['label'] = None

    if node.left:
        node_dict['left'] = build_cart_dict(node.left)
    else:
        node_dict['left'] = None

    if node.right:
        node_dict['right'] = build_cart_dict(node.right)
    else:
        node_dict['right'] = None

    node_dict['feature_idx'] = node.feature_idx

    return node_dict


def write_cart_tree(tree, path):

    tree_dict = {}
    meta_dict = {}
    id2node_dict = {}

    for k,v in tree.id2node.items():
        id2node_dict[int(k)] = build_cart_dict(v)

    tree_dict['id2node'] = id2node_dict

    # setup meta
    meta_dict['score'] = tree.score
    meta_dict['sample_size'] = int(tree.sample_size)
    meta_dict['lang_id'] = tree.lang_id

    tree_dict['meta'] = meta_dict

    json_str = json.dumps(tree_dict, indent=4)
    w = open(path, 'w', encoding='utf-8')
    w.write(json_str)
    w.close()


class Tree:

    def __init__(self, id2node, meta):

        self.meta = meta
        self.lang_id = self.meta['lang_id']
        self.score = self.meta['score']
        self.sample_size = self.meta['sample_size']
        self.id2node = id2node

        self.char = read_grapheme(self.lang_id)
        self.phoneme = read_inventory(self.lang_id).phoneme

    def __len__(self):

        return sum(len(node) for node in self.id2node.values())

    def add_top_node(self, center_id, node):
        self.id2node[center_id] = node


    def classify_units(self, chars):

        # map char to the closest char
        chars = [self.char.get_nearest_unit(char) for char in chars]

        ids = self.char.get_ids(chars)

        phoneme_id = self.classify_ids(np.array(ids))
        #print(ids, '->', phone_id)

        phoneme_lst = self.phoneme.get_joint_unit(phoneme_id)

        return phoneme_lst


    def classify_ids(self, x):

        center_id = x[2]
        if center_id in self.id2node:
            return self.id2node[center_id].classify(x)
        else:
            char = self.char.get_joint_unit(center_id)[0]
            phone = self.phoneme.get_nearest_phoneme(char)
            return self.phoneme.get_id(phone)


class Node:

    def __init__(self, config):
        self.config = config

        self.left_set = set()
        self.feature_idx = -1

        self.left = None
        self.right = None
        self.label = None

    def __len__(self):
        if self.label is not None:
            return 1
        else:
            return len(self.left) + len(self.right)

    def classify(self, x):

        if self.label is not None:
            return self.label

        if x[self.feature_idx] in self.left_set:
            #print("left")
            return self.left.classify(x)
        else:
            #print("right")
            return self.right.classify(x)
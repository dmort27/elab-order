import bz2
import math
from collections import defaultdict
import os
import epitran, panphon
import pandas as pd
import numpy as np
import torch
import gensim

class ElaborateExpressionData:
    def __init__(self, df, features='ton_rhy_ons', lang_id='', syl_regex=None, verbose=True,
                 remove_dup_ordered=None, wv_model=None):
        self.df = df
        self.lang_id = lang_id
        self.syl_regex = syl_regex
        self.verbose = verbose
        self.features = features
        # use only features from the coordinate compound for classification. may yield better result
        self.use_only_cc_words = True
        self._reformat_columns()
        self._remove_invalid_data()
        self._set_phonemes()
        self._add_unattested_data()
        if remove_dup_ordered is not None:
            self._remove_dups(remove_dup_ordered)
        self.X, self.y = None, None
        if 'wv' in features:
            assert wv_model is not None, "You must provide a source for the word vectors."
            self.wv = wv_model

    def rule_based_classification(self):

        if self.lang_id == 'hmn-Latn':
            orders = {'j': 1,
                      'b': 2,
                      'm': 3, 'd': 3,
                      's': 4,
                      'v': 5,
                      'g': 6,
                      '': 7}
            grp = 'ton'
        elif self.lang_id == 'lhu-Latn':
            orders = {'o': 1,
                      'u': 2,
                      'i': 3,
                      'ɨ': 4,
                      'ə': 5,
                      'ɔ': 6,
                      'e': 7,
                      'ɛ': 8,
                      'a': 9}
            grp = 'rhy'
        elif self.lang_id == 'ltc-IPA':
            orders = {'': 1,
                      'X': 2,
                      'H': 3,
                      'p̚':4, 't̚': 4, 'k̚': 4}
            grp = 'ton'
        else:
            print(f"Rule for language {self.lang_id} is not available")
            return -1
        rule_pred = (self.df['cc2'].apply(lambda syl: orders[self.syl_regex.match(syl).group(grp)]) -
                     self.df['cc1'].apply(lambda syl: orders[self.syl_regex.match(syl).group(grp)]))
        rule_pred += np.random.randn(len(rule_pred)) / 1000  # break ties randomly
        res = ((rule_pred>0) == self.df['attested']).value_counts().to_dict()

        N = len(self.df)
        if self.verbose:
            # print(f'Correct: {res[1] / N}')
            # print(f'Tie: {res[0] / N}')
            # print(f'Incorrect: {res[-1] / N}')
            print(f'Correct with random guess: {res[True] / N }')
            print(f'Incorrect with random guess: {res[False] / N }')
            print()
        return res[True] / N

    def rule_search(self):
        def sgn(x):
            return 0 if x == 0 else x // abs(x)
        from itertools import permutations
        N = len(self.df)
        best = 0
        if self.lang_id == 'hmn-Latn':
            permuted = list('jbmdsvg')+['']
            group = 'ton'
        elif self.lang_id == 'lhu-Latn':
            permuted = "iɨueəoɛaɔ"
            group = 'rhy'
        else:
            raise NotImplementedError

        perm = permutations(permuted)
        for iter in range(math.factorial(len(permuted))):
            p = next(perm)
            orders = {p[i]: i for i in range(len(permuted))}
            rule_pred = (self.df['cc2'].apply(lambda syl: orders[self.syl_regex.match(syl).group(group)]) -
                         self.df['cc1'].apply(lambda syl: orders[self.syl_regex.match(syl).group(group)]))
            res = rule_pred.apply(sgn).value_counts().to_dict()
            correct = res[1] / N + res[0] / N / 2
            if correct > best:
                best = correct
                print(f'iter {iter}, best is {best}, order is {orders}')

    def get_Xy_data(self):
        print('N =', len(self.df))
        if self.X is None and self.y is None:
            self._add_features(drop_orig_columns=True)
            self.y = self.df['attested'].to_numpy()
            self.df.drop(columns=['attested'], inplace=True)
            self.X = self.df.to_numpy()
            self.X_feature_names = self.df.columns.tolist()

        return self.X, self.y, self.X_feature_names, self.orig_index


    def _reformat_columns(self):
        '''
        Converts df from (word1, word2, word3, word4) into (cc1, cc2, rep, is_ABAC)
        '''
        if self.df.columns.tolist() == ['cc1', 'cc2', 'rep', 'is_ABAC']:
            if self.lang_id in ['ltc-IPA', 'cmn-Pinyin']:
                # rep has the characters, not needed for classification.
                self.df['rep'] = ''
                self.df['is_ABAC'] = True
            return
        if self.df.columns.tolist() != ['word1', 'word2', 'word3', 'word4']:
            print('Cannot reformat df. Need to be in (word1, word2, word3, word4) format')
            return

        df_ABAC = self.df[self.df.word1 == self.df.word3]
        df_ABAC = df_ABAC.rename(columns={'word1': 'rep', 'word2': 'cc1', 'word4': 'cc2'}).drop(columns='word3')
        df_ABAC['is_ABAC'] = True
        df_ABAC = df_ABAC[['cc1', 'cc2', 'rep', 'is_ABAC']]

        df_ABCB = self.df[self.df.word2 == self.df.word4]
        df_ABCB = df_ABCB.rename(columns={'word1': 'cc1', 'word2': 'rep', 'word3': 'cc2'}).drop(columns='word4')
        df_ABCB['is_ABAC'] = False
        df_ABCB = df_ABCB[['cc1', 'cc2', 'rep', 'is_ABAC']]

        self.df = pd.concat([df_ABAC, df_ABCB], names=['cc1', 'cc2', 'rep', 'is_ABAC'])

    def _remove_invalid_data(self):
        '''
        Removes elaborate expressions in df which cannot be parsed by syl_regex
        '''
        if self.df.columns.tolist() != ['cc1', 'cc2', 'rep', 'is_ABAC']:
            print('Cannot remove invalid data. Need to be in (cc1, cc2, rep, is_ABAC) format')
            return

        def is_valid_syl(syl):
            if syl == '':
                return True
            m = self.syl_regex.match(syl)
            if m:
                ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")
                return ons + rhy + ton == syl
            return False

        self.df.dropna(inplace=True)
        c = 0
        for i, (cc1, cc2, rep, form) in self.df.iterrows():
            if not all(is_valid_syl(w) for w in (cc1, cc2, rep)):
                self.df.drop(i, inplace=True)
                if self.verbose:
                    print(f"Dropping {i} because one of ({cc1}, {cc2}, {rep}) cannot be parsed")
            else:
                c += 1
        if self.verbose:
            print(c, 'good elaborate expressions left')
        self.df.reset_index(drop=True, inplace=True)

    def _set_phonemes(self):
        all_syllables = set(self.df["cc1"].tolist()).union(set(self.df["cc2"].tolist())).union(set(self.df["rep"].tolist()))
        all_syllables -= {''}
        self.onsets, self.rhymes, self.tones = set([]), set([]), set([])
        for syl in all_syllables:
            m = self.syl_regex.match(syl)
            ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")
            self.onsets.add(ons)
            self.rhymes.add(rhy)
            self.tones.add(ton)

    def _remove_dups(self, ordered):
        use_indices = []
        # remove duplicates in attested and unattested parts separately
        for partial_df in (self.df[self.df.attested], self.df[~self.df.attested]):
            inverted_idx = defaultdict(list)
            if ordered:
                for i, row in partial_df.iterrows():
                    cc1, cc2 = row['cc1'], row['cc2']
                    inverted_idx[(cc1, cc2)].append(i)
            else:
                for i, row in partial_df.iterrows():
                    cc1, cc2 = row['cc1'], row['cc2']
                    inverted_idx[(max(cc1, cc2), min(cc1, cc2))].append(i)
            for values in inverted_idx.values():
                use_indices.append(np.random.choice(values))

        self.df = self.df.iloc[use_indices].reset_index(drop=True)
        self.orig_index = self.orig_index[use_indices]

    def _add_unattested_data(self):
        '''
        for each (cc1, cc2) phrase, add (cc2, cc1) phrase
        '''
        unique_order_indices = []
        for i, (cc1, cc2, rep, form) in self.df.iterrows():
            other_order = self.df[(self.df.rep == rep) & (self.df.cc1 == cc2) & (self.df.cc2 == cc1)]
            if len(other_order) == 0:
                unique_order_indices.append(i)
        self.df['attested'] = True
        unattested = self.df.rename(columns={'cc1': 'cc2', 'cc2': 'cc1'}).iloc[unique_order_indices]
        unattested['attested'] = False
        if self.verbose:
            print(f'len of attested {len(self.df)}, len of unattested {len(unattested)}')

        # when combining, keep the original index for GroupKFold evaluation
        self.df = self.df.append(unattested, ignore_index=False).reset_index()
        self.orig_index = self.df.pop('index').to_numpy()  # store this orig_index elsewhere

    def _add_features(self, drop_orig_columns=True):
        features = self.features
        if 'panphon' in features:
            epi = epitran.Epitran(self.lang_id)
            ft = panphon.FeatureTable()

        if self.lang_id in ['ltc-IPA', 'cmn-Pinyin'] or self.use_only_cc_words:
            components = ('cc1', 'cc2')
        else:
            components = ('cc1', 'cc2', 'rep')
        # for lahu only, visualize tone marks on a dummy character 'a'
        dummy = 'a' if self.lang_id == 'lhu-Latn' else ''
        for wi in components:
            if 'ton' in features:
                for ton in self.tones:
                    self.df[f'{wi}_ton_{dummy}{ton}'] = self.df[wi].apply(lambda syl: self.syl_regex.match(syl).group("ton") == ton)
            if 'rhy' in features:
                for rhy in self.rhymes:
                    self.df[f'{wi}_rhy_{rhy}'] = self.df[wi].apply(lambda syl: self.syl_regex.match(syl).group("rhy") == rhy)
            if 'ons' in features:
                for ons in self.onsets:
                    self.df[f'{wi}_ons_{ons}'] = self.df[wi].apply(lambda syl: self.syl_regex.match(syl).group("ons") == ons)
            if 'panphon' in features:
                wordi_feats = self.df[wi].apply(lambda syl: ft.bag_of_features(epi.transliterate(syl)))
                ## TODO: panphone feature for ons and rhy separately?
                panphon_names = [f'{wi}_{sign}{n}' for n in ft.names for sign in ('+', '0', '-')]
                self.df = pd.merge(
                        self.df,
                        pd.DataFrame(wordi_feats.tolist(), index=self.df.index, columns=panphon_names),
                        left_index=True, right_index=True)
            if 'wv' in features:
                wv_feats = self.df[wi].apply(lambda syl: self.wv[syl])
                col_names = [f'{wi}_wv{i}' for i in range(self.wv.get_wv_dim())]
                self.df = pd.concat((self.df, pd.DataFrame(wv_feats.tolist(), columns=col_names)), axis=1)


        if drop_orig_columns:
            self.df.drop(columns=['cc1', 'cc2', 'rep'], inplace=True)

    def _add_wv_features_to_X(self):
        if 'wv' not in self.features:
            return


    def get_feature_names(self):
        ret = self.df.columns.to_list()
        ret.remove('attested')
        return ret



class WordVectorsData:
    def __init__(self, model_name):
        self.model_name = model_name
        self.vector_size = 0
        self.get_wv = None

    def __getitem__(self, word):
        try:
            return self.get_wv(word)
        except KeyError:
            return np.zeros(self.get_wv_dim())
        # return np.random.random(self.get_wv_dim())

    def get_wv_dim(self):
        return self.vector_size

class HmongWordVectorsData(WordVectorsData):
    def __init__(self, model_name):
        super().__init__(model_name)
        SERVER = "/run/user/1000/gvfs/sftp:host={}.lti.cs.cmu.edu,user=cxcui/usr0/home/cxcui/pytorch-NER/model/"

        if model_name in ('sg', 'cbow'):
            wv = gensim.models.Word2Vec.load(f"../data/hmong/sch.{model_name}.w2v").wv
            self.vector_size = wv.vector_size
            self.get_wv = wv.__getitem__
            return

        for server_name in ('agent', 'patient'):
            server_base = SERVER.format(server_name)
            if not os.path.exists(server_base):
                print("WARNING: server file system not mounted")
            path = os.path.join(server_base, model_name, 'best_model')
            if os.path.exists(path):
                self.embeds = torch.load(path, map_location='cpu')['embeds.weight']
                self.w2i = torch.load("../data/hmong/data.pth")['w2i']
                self.vector_size = self.embeds.shape[1]
                self.get_wv = lambda w: self.embeds[self.w2i.get(w, self.w2i.get('UNK'))].numpy()
                print(f"loaded word vectors from {model_name} on {server_name}")
                break
        else: # no break
            raise FileNotFoundError(f"Cannot find a wv model with name {model_name}")

class ChineseWordVectorsData(WordVectorsData):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.get_wv = self.get_chinese_wv().__getitem__

    def get_chinese_wv(self):
        wv_file_path = f"/home/cuichenx/Downloads/{self.model_name}.bz2"
        out_name = f"../data/chinese/{self.model_name}_selected.wv"
        char2wv = {}

        if not os.path.exists(out_name):
            words_of_interest = pd.read_csv("../data/chinese/extracted_coordinate_compounds/pinyin.csv")['rep'].tolist()
            chars_of_interest = set(''.join(words_of_interest))

            with bz2.BZ2File(wv_file_path) as f:
                for i, line in enumerate(f):
                    word = line.split(b' ', maxsplit=1)[0].decode()
                    # print(word, len(word))
                    if len(word) == 1 and word in chars_of_interest:
                        chars_of_interest.remove(word)
                        wv_str = line.split(b' ', maxsplit=1)[1].decode()
                        char2wv[word] = wv_str

                        print(f"found {word}, {len(chars_of_interest)} characters of interest left")
                        if len(chars_of_interest) == 0:
                            break

            with open(out_name, 'w') as f:
                for word, wv_str in char2wv.items():
                    f.write(f"{word} {wv_str}")
        else:
            print("using existing:", out_name)

        char2wv = {}
        with open(out_name) as f:
            for line in f:
                word, wv = line.split(' ', maxsplit=1)
                char2wv[word] = np.fromstring(wv, sep=' ')
            self.vector_size = len(wv)
        return char2wv
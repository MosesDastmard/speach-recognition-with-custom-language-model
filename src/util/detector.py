from joblib import Parallel, delayed
import pulp as plp
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import sys
MISSED_CHAR = "?"

class Detector:
    def __init__(self) -> None:
        self.mapping = None

    @staticmethod
    def extend(a, b):
        len_dif = len(a) - len(b)
        if len_dif > 0:
            b += MISSED_CHAR * len_dif
        if len_dif < 0:
            a += MISSED_CHAR * (-len_dif)
        return a, b


    def get_extention(self, reference_sentence, predicted_sentence):
        reference_sentence, predicted_sentence = self.extend(reference_sentence, predicted_sentence)
        if len(reference_sentence) == 0:
            return reference_sentence, predicted_sentence
        I = range(len(reference_sentence))

        X = {(i, j): plp.LpVariable(f'X_{i}_{j}', cat="Binary") for i in I for j in I}

        error_model = plp.LpProblem(name="MILP_Model")

        # connection can exist if two char are the same
        [error_model.addConstraint(plp.LpConstraint(
            e=X[(i, j)],
            sense=plp.LpConstraintLE,
            rhs=1 if reference_sentence[i] == predicted_sentence[j] else 0
        ))
            for i in I for j in I
        ];

        # each char in a can assing to only one char in b or nothing
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)] for i in I]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for j in I
        ];

        # each char in b can assing to only one char in a or nothing
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)] for j in I]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for i in I
        ];

        # avoid cross connections clockwise
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)], X[ip, jm]]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for i in I for j in I for ip in I if ip > i for jm in I if jm < j
        ];

        # avoid cross connections counter-clockwise
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)], X[im, jp]]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for i in I for j in I for im in I if im < i for jp in I if jp > j
        ]

        objective = plp.lpSum([X[(i, j)] for i in I for j in I])
        error_model.sense = plp.LpMaximize
        error_model.setObjective(objective)
        solver = plp.apis.coin_api.PULP_CBC_CMD(msg=False)
        error_model.solve(solver)

        x_df = pd.DataFrame.from_dict(X, orient="index",
                                    columns=["variable_object"])
        x_df["solution_value"] = x_df["variable_object"].apply(lambda item: item.varValue)

        sol_df = x_df[x_df['solution_value'] > 0]
        a_dic = {i: c for i, c in enumerate(reference_sentence)}
        b_dic = {i: c for i, c in enumerate(predicted_sentence)}
        for (i, j), _ in sol_df.iterrows():
            if i == j:
                continue
            elif i > j:
                tmp_dic = dict()
                for p, c in a_dic.items():
                    if p <= i:
                        tmp_dic[p - i + j] = c
                    else:
                        tmp_dic[p] = c
                a_dic = tmp_dic
                tmp_dic = dict()
                for p, c in b_dic.items():
                    if p < j:
                        tmp_dic[p - i + j] = c
                    else:
                        tmp_dic[p] = c
                b_dic = tmp_dic
            elif i < j:
                tmp_dic = dict()
                for p, c in a_dic.items():
                    if p < i:
                        tmp_dic[p - j + i] = c
                    else:
                        tmp_dic[p] = c
                a_dic = tmp_dic
                tmp_dic = dict()
                for p, c in b_dic.items():
                    if p <= j:
                        tmp_dic[p - j + i] = c
                    else:
                        tmp_dic[p] = c
                b_dic = tmp_dic
        min_ind = min(list(a_dic.keys()) + list(b_dic.keys()));
        max_ind = max(list(a_dic.keys()) + list(b_dic.keys()));
        for p in range(min_ind, max_ind + 1):
            if p not in a_dic.keys():
                a_dic[p] = MISSED_CHAR
            if p not in b_dic.keys():
                b_dic[p] = MISSED_CHAR
        b_ext = "".join([b_dic[p] for p in range(min_ind, max_ind + 1) if
                        not ((b_dic[p] == MISSED_CHAR) and (a_dic[p] == MISSED_CHAR))])
        a_ext = "".join([a_dic[p] for p in range(min_ind, max_ind + 1) if
                        not ((b_dic[p] == MISSED_CHAR) and (a_dic[p] == MISSED_CHAR))])
        return a_ext, b_ext


    def find_error(self, a, b):
        a, b = self.extend(a, b)
        default_win_stride = [[3, 2], [6, 2], [7, 3], [8, 3], [9, 5]]
        # win_stride = [w_s for w_s in default_win_stride if w_s[0] > len(a)]
        sol = dict()
        for Y, (w, s) in enumerate(default_win_stride):
            window_size = w
            stride = s
            a_cor = a
            b_cor = b
            start = 0
            while (start + window_size) < len(a_cor) and (start + window_size) < len(b_cor):
                a_fin = a_cor[:start]
                b_fin = b_cor[:start]
                a_win = a_cor[start:(start + window_size)]
                a_res = a_cor[(start + window_size):]
                b_win = b_cor[start:(start + window_size)]
                b_res = b_cor[(start + window_size):]
                a_ext, b_ext = self.get_extention(a_win, b_win)
                a_cor = a_fin + a_ext + a_res
                b_cor = b_fin + b_ext + b_res
                start += stride
            a_fin = a_cor[:start]
            b_fin = b_cor[:start]
            a_win = a_cor[start:]
            b_win = b_cor[start:]
            a_ext, b_ext = self.get_extention(a_win, b_win)
            a_cor = a_fin + a_ext
            b_cor = b_fin + b_ext
            count = 0
            for i in range(len(a)):
                if a_cor[i] == b_cor[i]:
                    count += 1
            score = count / len(a)
            sol[Y] = {'score': score, 'a_cor': a_cor, 'b_cor': b_cor}
        return sol


    def get_error(self, a, b):
        sol = self.find_error(a, b)
        score = 0
        key = None
        for i, info in sol.items():
            if info['score'] > score:
                key = i
                score = info['score']
        if key is not None:
            return sol[key]

        return {
                "a_cor":" ", 
                "b_cor":" "
                }


    def get_map(self, a, b):
        a_cor, b_cor = self.get_pairs(a, b)
        return zip(a_cor, b_cor)


    def get_maps(self, a_list, b_list):
        return [list(self.get_map(a, b)) for a, b in zip(a_list, b_list)]


    def get_pairs(self, a, b):
        sol = self.get_error(a, b)
        a_cor = sol['a_cor']
        b_cor = sol['b_cor']
        return a_cor, b_cor

    @staticmethod
    def get_ngram_error(a, b, n):
        a_wins = []
        b_wins = []
        for i in range(len(a) - n + 1):
            a_win = a[i:(i + n)]
            b_win = b[i:(i + n)]
            a_wins.append(a_win)
            b_wins.append(b_win)
        return a_wins, b_wins

    def train(self, actual_sentences, predicted_sentences, order, model_path):
        # actual_sentences = actual_sentences[0:10]
        # predicted_sentences = predicted_sentences[0:10]
        def get_mapping(actual_sentence, predicted_sentence):
            a_, b_ = self.get_pairs(actual_sentence, predicted_sentence)
            pair = []
            for o in range(1, order+1):
                a__, b__ = self.get_ngram_error(a_, b_, o)
                for i,j in zip(a__, b__):
                    if i != MISSED_CHAR:
                        i_ = i.replace(MISSED_CHAR, "")
                        j_ = j.replace(MISSED_CHAR, "")
                        pair.append((i_, j_))
            return pair
        jobs = []
        for actual_sentence, predicted_sentence in tqdm(zip(actual_sentences, predicted_sentences)):
            jobs.append(delayed(get_mapping)(actual_sentence, predicted_sentence))

        pairs = Parallel(n_jobs=16, verbose=1)(jobs)
        mapping = dict()
        for pair in pairs:
            for i_,j_ in pair:
                if i_ not in mapping.keys():
                    mapping[i_] = [j_]
                else:
                    mapping[i_].append(j_)

        json.dump(mapping, open(model_path, 'w'))
        self.mapping = mapping

    @staticmethod
    def map_chars(chunk, mapping):
        if chunk in mapping.keys():
            candidates = mapping[chunk]
            print(chunk, candidates)
            rnd = np.random.randint(len(candidates))
            chunk_out = candidates[rnd]
        else:
            chunk_out = chunk
        # print(chunk, chunk_out)
        return np.array(chunk_out, dtype='<U3')

    def load(self, model_path):
        mapping = json.load(open(model_path, 'r'))
        def mapper(text):
            text_len = len(text)
            # p=np.array([2.95e-5, 0.00088, 0.01344, 0.12629, 0.8593])
            # p = p/np.sum(p)
            indices = np.random.choice([1,2,3,4], text_len, replace=True)
            # indices = np.random.randint(low=1, high=5, size=text_len)
            indices = np.concatenate([[0], indices])
            indices = np.cumsum(indices)
            indices = indices[indices < text_len]
            indices = np.concatenate([indices, [text_len]])
            indices = np.array([indices[0:-1], indices[1:]])
            augmented_text = np.apply_along_axis(func1d=lambda idx: self.map_chars(text[idx[0]:idx[1]], mapping), arr=indices, axis=0)
            augmented_text = "".join(augmented_text.tolist())
            return augmented_text
        
        return mapper
     


        
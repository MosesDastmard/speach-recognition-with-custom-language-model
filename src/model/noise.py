from joblib import Parallel, delayed
import pulp as plp
import pandas as pd
import numpy as np
from src.dataset.loader import Loader
from tqdm import tqdm
from src.util import config
from src.util.functions import match_text
import numpy as np
import os
import json


class util:
    @staticmethod
    def __align(reference_sentence : str, corrupted_sentence : str) -> tuple[str, str]:
        """It tries to align a grandtruth sentence to a given predicted sentence with lowest
        error levels. In case of deletion or insertion, it uses NULL_CHAR to fill the blanks
        and makes two sentences match each other with higherst same characters.

        Args:
            reference_sentence (str): clean sentence
            corrupted_sentence (str): predicted sentence which may or may not contain errors

        Returns:
            tuple[str, str]: (processed_clean_sentence, processed_predicted_sentence) 
            processed_clean_sentence is clean sentence with NULL_CHAR so that it has highest
            similarity to processed_predicted_sentence (it has NULL_CHAR as well). 
            processed_predicted_sentence and processed_clean_sentence are always of the same
            length.
        """
        # padding with NULL_CHAR so that reference_sentence and predicted_sentence have the same length
        reference_sentence, corrupted_sentence = match_text(reference_sentence, corrupted_sentence)
        # list index of characters in reference_sentence
        I = range(len(reference_sentence))
        # variable which shows the connection between reference_sentence's characters 
        # and predicted_sentence's characters
        X = {(i, j): plp.LpVariable(f'X_{i}_{j}', cat="Binary") for i in I for j in I if np.abs(i-j) < 10}
        # Initialize MLP model
        error_model = plp.LpProblem(name="MILP_Model")
        
        ##################### constraints #####################
        # connection can exist if two chars are the same
        [error_model.addConstraint(plp.LpConstraint(
            e=X[(i, j)],
            sense=plp.LpConstraintLE,
            rhs=1 if reference_sentence[i] == corrupted_sentence[j] else 0
        ))
            for i in I for j in I if (i,j) in X.keys()
        ];

        # each char in text_a can assing to only one char in text_b or nothing
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)] for i in I if (i,j) in X.keys()]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for j in I
        ];

        # each char in text_b can assing to only one char in text_a or nothing
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)] for j in I if (i,j) in X.keys()]),
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
            for i in I for j in I if (i,j) in X.keys()
            for ip in I if (ip > i) for jm in I if (jm < j) and (ip,jm) in X.keys()
        ];

        # avoid cross connections counter-clockwise
        [error_model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[(i, j)], X[im, jp]]),
            sense=plp.LpConstraintLE,
            rhs=1
        ))
            for i in I for j in I if (i,j) in X.keys()
            for im in I if im < i for jp in I if (jp > j) and (im,jp) in X.keys()
        ]

        # objective: sum of connections which is aimed to minimize
        objective = plp.lpSum([X[(i, j)] for i in I for j in I if (i,j) in X.keys()])
        error_model.sense = plp.LpMaximize
        error_model.setObjective(objective)
        # solve the problem
        solver = plp.apis.coin_api.PULP_CBC_CMD(msg=False)
        error_model.solve(solver)

        # extract solutions into pd.DataFrame
        x_df = pd.DataFrame.from_dict(X, orient="index",
                                    columns=["variable_object"])
        x_df["solution_value"] = x_df["variable_object"].apply(lambda item: item.varValue)
        df = pd.DataFrame({'reference_sentence':list(reference_sentence),
                    "predicted_sentence": list(corrupted_sentence),
                    "index": range(len(corrupted_sentence))})

        # we only care about those characters that are connected
        sol_df = x_df[x_df['solution_value'] > 0]
        
        ########################## process solution #########################
        """This section applies the solution to the reference sentence and 
        predicted sentence. It adds NULL_CHAR at proper positions
        """        
        text_a_dic = {i: c for i, c in enumerate(reference_sentence)}
        text_b_dic = {i: c for i, c in enumerate(corrupted_sentence)}
        for (i, j), _ in sol_df.iterrows():
            # if two chars has connection at same positions, process is not necessary
            if i == j:
                continue
            # if text_a has a character which is connected to text_b char with higher position,
            # in text_b NULL_CHAR must be inserted to correct positions
            elif i > j:
                tmp_dic = dict()
                for p, c in text_a_dic.items():
                    if p <= i:
                        tmp_dic[p - i + j] = c
                    else:
                        tmp_dic[p] = c
                text_a_dic = tmp_dic
                tmp_dic = dict()
                for p, c in text_b_dic.items():
                    if p < j:
                        tmp_dic[p - i + j] = c
                    else:
                        tmp_dic[p] = c
                text_b_dic = tmp_dic
            # if text_b has a character which is connected to text_a char with higher position,
            # in text_a NULL_CHAR must be inserted to correct positions
            elif i < j:
                tmp_dic = dict()
                for p, c in text_a_dic.items():
                    if p < i:
                        tmp_dic[p - j + i] = c
                    else:
                        tmp_dic[p] = c
                text_a_dic = tmp_dic
                tmp_dic = dict()
                for p, c in text_b_dic.items():
                    if p <= j:
                        tmp_dic[p - j + i] = c
                    else:
                        tmp_dic[p] = c
                text_b_dic = tmp_dic
        min_ind = min(list(text_a_dic.keys()) + list(text_b_dic.keys()));
        max_ind = max(list(text_a_dic.keys()) + list(text_b_dic.keys()));
        for p in range(min_ind, max_ind + 1):
            if p not in text_a_dic.keys():
                text_a_dic[p] = config.NULL_CHAR
            if p not in text_b_dic.keys():
                text_b_dic[p] = config.NULL_CHAR
        text_b_ext = "".join([text_b_dic[p] for p in range(min_ind, max_ind + 1) if
                        not ((text_b_dic[p] == config.NULL_CHAR) and (text_a_dic[p] == config.NULL_CHAR))])
        text_a_ext = "".join([text_a_dic[p] for p in range(min_ind, max_ind + 1) if
                        not ((text_b_dic[p] == config.NULL_CHAR) and (text_a_dic[p] == config.NULL_CHAR))])
        
        
        return text_a_ext, text_b_ext

    @staticmethod
    def __get_alignment(reference_sentence : str, corrupted_sentence : str) -> tuple[str, str]:        
        """It is an wrapper for util.__align() function, it makes sure that the two sentences
        are fine to fire util.__align() function on them

        Args:
            reference_sentence (str): clean sentence
            corrupted_sentence (str): predicted sentence which may or may not contain errors

        tuple[str, str]: (processed_clean_sentence, processed_predicted_sentence) 
            processed_clean_sentence is clean sentence with NULL_CHAR so that it has highest
            similarity to processed_predicted_sentence (it has NULL_CHAR as well). 
            processed_predicted_sentence and processed_clean_sentence are always of the same
            length. 
        """
        # check of sentence are not empty    
        if (reference_sentence != "") and (corrupted_sentence != ""):
            # in case two sentences are the same, firing util.__align() function
            # is not necessary
            if reference_sentence == corrupted_sentence:
                text_a_ext, text_b_ext = reference_sentence, corrupted_sentence
                return text_a_ext, text_b_ext
            else:
            # if two sentences have length different percentage more than 80%
            # it skip firing util.__align() function because it is likely that
            # the sentences too different.
                ref_len = len(reference_sentence)
                pre_len = len(corrupted_sentence)
                criteria = np.abs(ref_len - pre_len)/np.max([ref_len, pre_len])
                if criteria < 0.8:
                    text_a_ext, text_b_ext = util.__align(reference_sentence, corrupted_sentence)
                    return text_a_ext, text_b_ext
                else:
                    print("too diffrent sentences exist")
                    print("reference_sentence:", reference_sentence)
                    print("predicted_sentence:", corrupted_sentence)        
        else:
            print("null sentence exists")
            print("reference_sentence:", reference_sentence)
            print("predicted_sentence:", corrupted_sentence)
        return None

    @staticmethod
    def get_alignments(reference_sentences : list[str], corrupted_senteneces : list[str], purifier:function=lambda x:x) -> tuple[list[str], list[str]]:
        """It aligns the reference sentences and corrupted sentences, it's quit same as util.__get_alignment() method,
        but it works on list of sentences.

        Args:
            reference_sentences (list[str]): clean sentences
            corrupted_senteneces (list[str]): corrupted sentences
            purifier (function, optional): function that process sentences before feeding them firing aligment. Defaults to lambdax:x.

        Raises:
            Exception: both reference_sentences and corrupted_senteneces has to be the same length

        Returns:
            tuple[list[str], list[str]]: processed_reference_sentences, processed_corrupted_senteneces
        """              
        if not all(map(lambda x: isinstance(x, str), reference_sentences + corrupted_senteneces)):
            raise Exception(f"all items in reference_sentence and predicted_senteneces must be of type str")
        jobs = []
        for reference_sentence, predicted_sentence in tqdm(zip(reference_sentences, corrupted_senteneces)):
            jobs.append(delayed(util.__get_alignment)(purifier(reference_sentence), purifier(predicted_sentence)))
        pairs = Parallel(n_jobs=os.cpu_count(), verbose=10)(jobs)
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    @staticmethod
    def get_reverse_mapper(reference_sentences : list[str], corrupted_sentences:list[str], max_gram:int, reverse_mapper:dict=dict()) -> dict:
        """It makes chunks of characters of size 1 to `max_gram` characters so that for a given character chunks in
        a corrupted sentences, it makes a list of all the matched chunk of characters in the reference sentence.
        two corresponding chunks have the same positions in both reference sentence and corrupted sentence.

        Args:
            reference_sentences (list[str]): clean sentences
            corrupted_sentences (list[str]): corrupted sentences
            max_gram (int): max size to make characters chunk
            reverse_mapper (dict, optional): _description_. Defaults to dict().

        Raises:
            Exception: reference sentence and corrupted sentence must have the same length

        Returns:
            dict: dictionary that keys are chunks from corrrupted sentences without NULL_CHAR and corresponding
            list of possible clean chunks taken from the clean sentences.
        """        
        for ref, cor in zip(reference_sentences, corrupted_sentences):
            if len(ref) == len(cor):
                for j in range(1, max_gram):
                    for i in range(j, len(ref)+1):
                        ref_win = ref[i-j:i].replace(config.NULL_CHAR, "")
                        pre_win = cor[i-j:i].replace(config.NULL_CHAR, "")
                        if pre_win != "":
                            if pre_win in reverse_mapper.keys():
                                reverse_mapper[pre_win].append(ref_win)
                            else:
                                reverse_mapper[pre_win] = [ref_win]
            else:
                raise Exception("reference sentence and corrupted sentence must have the same length.")
        return reverse_mapper
    
    
class Ngram:
    def __init__(self, n=100, max_gram=12):
        self.max_gram = max_gram
        self.n = n
        self.__reverse_mapper = dict()

    def suggestions(self, corrupted_sentence):
        if self.__reverse_mapper == dict():
            raise Exception("model is not initialized yet. Try .fit or .load to initialize the model.")
        jobs = [delayed(self.__suggestion)(corrupted_sentence, self.__reverse_mapper, self.max_gram) for _ in range(self.n)]
        suggested_sentences = Parallel(n_jobs=os.cpu_count())(jobs)
        return suggested_sentences
    
    @staticmethod
    def __suggestion(corrupted_sentence, reverse_mapper, max_gram):
        processed_sequence = []
        window_sequence = ""
        remain_sequence = corrupted_sentence
        attempts = 0
        max_attemepts = 200
        while True:
            attempts += 1
            if len(remain_sequence) == 0:
                break
            win_size = np.random.randint(low=1, high=max_gram)
            window_sequence = remain_sequence[0:win_size]
            if window_sequence in reverse_mapper.keys():
                processed_sequence.append([window_sequence, reverse_mapper[window_sequence]])
                remain_sequence = remain_sequence[win_size:]
            elif attempts > max_attemepts:
                processed_sequence.append([window_sequence, [window_sequence]])
                remain_sequence = remain_sequence[win_size:]
                print(f"Warnning, noise model can't randomizied the sentence after {max_attemepts} tries. You may train the model on samll dataset")
        return "".join([np.random.choice(x[1]) for x in processed_sequence])
         
        
    def fit(self, reference_sentences, corrupted_sentences, purifier=lambda x:x):
        reference_sentences, corrupted_sentences = util.get_alignments(reference_sentences, corrupted_sentences, purifier)
        self.__reverse_mapper = util.get_reverse_mapper(reference_sentences, corrupted_sentences, self.max_gram, self.__reverse_mapper)
        return self

    def save(self, model_path):
        with open(model_path, 'w') as f:
            json.dump(self.__reverse_mapper, f)

    def load(self, model_path):
        with open(model_path, 'r') as f:
            self.__reverse_mapper = json.load(f)
        return self




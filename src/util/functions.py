from . import config
import re
import subprocess
from nltk import word_tokenize
import numpy as np
import pulp as plp
import pandas as pd
class Process:
    def __init__(self, command) -> None:
        self.command = command

    def run(self):
        subprocess.run(self.command, shell=True, check=True)


def purify_text(text):
    text = text.lower()
    regex = ",".join(list(config.CHAR_SET))
    regex = fr'[{regex}]'
    pure_text = "".join(re.findall(regex, text)).lower()
    pure_text = re.sub(pattern=config.IGNORE_PUNC, repl="", string=pure_text)
    return pure_text.strip()

def shrink(text):
    stride = config.STRIDE
    texts = []
    if len(text) >= config.MAX_CHAR:
        start = 0
        end = start + config.MAX_CHAR
        while end <= len(text):
            subtext = text[start:end]
            start += stride
            end += stride
            texts.append(subtext)
        subtext = text[start:]
        texts.append(subtext)
        return texts
    else:
        return [text]



max_allowed_tokens = int(config.MAX_TOKEN_INPUT*.8)

def split_sentence(sentence, n):
    if n < 1:
        raise Exception(f"n has to be a positive number. It's {n}")
    words = word_tokenize(sentence, language='italian')
    words_len = len(words)
    if n > words_len:
        raise Exception(f"There are {words_len} words in '{sentence}'. n (which is set to {n}) has to be less or equal to number of words ({words_len})")
    N = range(n)
    X = {i: plp.LpVariable(f'X_{i}', cat="Integer") for i in N}
    Z = plp.LpVariable('Z', cat="Integer")
    model = plp.LpProblem(name="MILP_Model")
    model.addConstraint(plp.LpConstraint(
            e=plp.lpSum([X[i] for i in N]),
            sense=plp.LpConstraintEQ,
            rhs=words_len
        ))
    for i in N:
        model.addConstraint(plp.LpConstraint(
            e= Z - X[i],
            sense=plp.LpConstraintLE,
            rhs=0,
        ))

    model.sense = plp.LpMaximize
    model.setObjective(Z)
    solver = plp.apis.coin_api.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    x_df = pd.DataFrame.from_dict(X, orient="index",
                                    columns=["variable_object"])
    x_df["solution_value"] = x_df["variable_object"].apply(lambda item: item.varValue)
    chunks = x_df['solution_value'].astype(int).tolist()
    small_sentences = []
    c = 0
    for l in chunks:
        small_sentnece = []
        for _ in range(l):
            small_sentnece.append(words[c])
            c += 1
        small_sentences.append(" ".join(small_sentnece))
    return small_sentences
        

class AutoSplitSentence:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokens_len(self, sentence):
        ids = self.tokenizer.tokenize(sentence)
        return len(ids)

    def get_max_token_len(self, sentences):
        lens = []
        for sentence in sentences:
            lens.append(self.get_tokens_len(sentence))
        return max(lens)

    def __call__(self, sentence):
        for n in range(1, 1+len(word_tokenize(sentence))):
            small_sentences = split_sentence(sentence, n)
            max_l = self.get_max_token_len(small_sentences)
            if max_l < max_allowed_tokens:
                break
        return small_sentences


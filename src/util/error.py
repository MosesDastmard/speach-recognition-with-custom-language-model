from src.util import config
from lib.parpandas.parallel_pandas import read_csv_parallel
import json
import numpy as np
from tqdm import tqdm

mapping = json.load(open(config.MAPPING_JSON_FILE, 'r'))

def map_chars(idx, chunk):
    if chunk in mapping.keys():
        chunk_out = np.random.choice(mapping[chunk])
    else:
        chunk_out = chunk
    # print(chunk, chunk_out)
    return np.array(chunk_out, dtype='<U3')

class Mapper:
    def __init__(self) -> None:
        pass

    def get_mapper(self):
        def mapper(text):
            text_len = len(text)
            indices = np.random.randint(low=1, high=4, size=text_len)
            indices = np.concatenate([[0], indices])
            indices = np.cumsum(indices)
            indices = indices[indices < text_len]
            indices = np.concatenate([indices, [text_len]])
            indices = np.array([indices[0:-1], indices[1:]])
            augmented_text = np.apply_along_axis(func1d=lambda idx: map_chars(idx, text[idx[0]:idx[1]]), arr=indices, axis=0)
            augmented_text = "".join(augmented_text.tolist())
            return augmented_text
        
        return mapper




class Session:
    def __init__(self) -> None:
        pass

    def run(self):
        df = read_csv_parallel(config.ERRORS_PATH, n_jobs=8, parallel_on_files=True)
        df = df.dropna().copy()
        df['ref_ngram'] = df['ref_ngram'].str.replace(config.SEPCIAL_CHAR, "")
        df['pre_ngram'] = df['pre_ngram'].str.replace(config.SEPCIAL_CHAR, "")
        mapping = dict()
        for ref_ngram, gr in tqdm(df.groupby('ref_ngram')):
            pre_list = [x.upper() for x in gr['pre_ngram'].tolist() if all([c.upper() in config.CHAR_SET for c in x])]
            if ref_ngram in pre_list and ref_ngram != "":
                mapping[ref_ngram] = pre_list
        json.dump(mapping, open(config.MAPPING_JSON_FILE, 'w'))


Session().run()


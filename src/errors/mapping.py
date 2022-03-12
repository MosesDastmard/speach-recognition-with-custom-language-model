from src.parpandas.parallel_pandas import read_csv_parallel
from src.util import config
from tqdm import tqdm
import json


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
from pyspark.sql import SparkSession
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import multiprocessing
import re
from src.util import config
import os
from argparse import ArgumentParser
from src.util.functions import purify_text
from src.util.detector import Detector
mapper = Detector().load(config.MAPPING_MODEL_PATH)
from src.tokenizer.bpe import BPE

if config.MODE == 'small':
    bpe_tokenizer = BPE.load(config.TOKENIZER_MODEL_SMALL_PATH)
else:
    bpe_tokenizer = BPE.load(config.TOKENIZER_MODEL_PATH)

def join_tokens(clean_sentence):
    corrupted_sentence = mapper(clean_sentence)
    tokens_clean = bpe_tokenizer.tokenize(clean_sentence, result='tokens')
    tokens_corrupted = bpe_tokenizer.tokenize(corrupted_sentence, result='tokens')
    mutual_len = max([len(tokens_clean), len(tokens_corrupted)])
    if mutual_len > len(tokens_clean):
        for _ in range(mutual_len-len(tokens_clean)):
            tokens_clean.insert(-1, '[PAD]')
    joint_tokens = "~".join(tokens_corrupted) + "|" + "~".join(tokens_clean)
    return joint_tokens

def filter(token_format_sentence):
    corrupted_token_format_sentence, clean_token_format_sentence = token_format_sentence.split("|")
    if corrupted_token_format_sentence.count("~") >= config.MAX_TOKEN_INPUT:
        return False
    if clean_token_format_sentence.count("~") >= config.MAX_TOKEN_INPUT:
        return False
    return True

def slice_sentence(sentence):
    window = 64
    stride = 32
    sentences = []
    if len(sentence) < window:
        sentences.append(sentence)
    else:
        start = 0
        end = window
        while end < len(sentence):
            sentence_ = sentence[start:end]
            sentences.append(sentence_)
            start += stride
            end += stride
        if start < len(sentence):
            sentence_ = sentence[start:]
            sentences.append(sentence_)
    return sentences

class Session:
    def __init__(self, input_text_file_path, output_text_file_path, mode) -> None:
        self.input_text_file_path = input_text_file_path
        self.output_text_file_path = output_text_file_path
        self.mode = mode
        
    def start_saprk_session(self):
        cores_number = multiprocessing.cpu_count()
        self.spark = SparkSession.builder.appName('dataset_builder').master(f'local[{cores_number}]').getOrCreate()
        print(f"spark runs on {cores_number} cores.")

    def get_partition_number(self):
        input_file_size = os.stat(self.input_text_file_path).st_size
        partition_number = int(input_file_size/config.PARTITION_SIZE)
        print(f'spark runs on {partition_number} partitions')
        return partition_number
        
    def compute(self):
        partition_number = self.get_partition_number()
        ref_rdd = self.spark.sparkContext.textFile(self.input_text_file_path, minPartitions=partition_number)
        if self.mode == 'preprocess':
            ref_rdd = ref_rdd.flatMap(sent_tokenize)
            ref_rdd = ref_rdd.map(purify_text)
            ref_rdd = ref_rdd.filter(lambda x: len(word_tokenize(x)) > 3)
        if self.mode == 'clean.corrupted':
            ref_rdd = ref_rdd.flatMap(slice_sentence)
            ref_rdd = ref_rdd.map(join_tokens)
            ref_rdd = ref_rdd.filter(filter)
        if self.mode == 'corrupted':
            ref_rdd = ref_rdd.map(mapper)
        os.system(f"rm -rf {self.output_text_file_path}")
        ref_rdd.saveAsTextFile(self.output_text_file_path)
        
        
    def merge(self):
        all_files = os.listdir(self.output_text_file_path)
        used_files = [file for file in all_files if file.startswith('part')]
        for file in all_files:
            if file not in used_files:
                os.remove(os.path.join(self.output_text_file_path, file))
        tmp_file_name = 'tmp.tmp.txt'
        tmp_file_path = os.path.join(self.output_text_file_path, tmp_file_name)
        print(tmp_file_path)
        os.system(f'cat {self.output_text_file_path}/* >> {tmp_file_path}')
        new_tmp_file_path = os.path.join(os.path.dirname(self.output_text_file_path), tmp_file_name)
        os.system(f'mv {tmp_file_path} {new_tmp_file_path}')
        os.system(f'rm -rf {self.output_text_file_path}')
        os.system(f'mv {new_tmp_file_path} {self.output_text_file_path}')
        
        
    def run(self):
        self.start_saprk_session()
        self.compute()
        if self.mode != "clean.corrupted":
            self.merge()

class Preprocess:
    def __init__(self, input_text_file=config.CC100_PATH, output_text_file=config.CC100_PREPROCESSED_PATH):
        if input_text_file is None:
            input_text_file=config.CC100_PATH
        if output_text_file is None:
            output_text_file = config.CC100_PREPROCESSED_PATH
        self.session = Session(input_text_file, output_text_file, 'preprocess')
    
    def run(self):
        self.session.run()


class CleanCorrupted:
    def __init__(self, input_text_file=config.CC100_PREPROCESSED_PATH, output_text_file=config.CC100_CLEAN_CORRUPTED_PATH):
        if input_text_file is None:
            input_text_file=config.CC100_PREPROCESSED_PATH
        if output_text_file is None:
            output_text_file = config.CC100_CLEAN_CORRUPTED_PATH
        self.session = Session(input_text_file, output_text_file, 'clean.corrupted')
    
    def run(self):
        self.session.run()


class Corrupted:
    def __init__(self, input_text_file=config.CC100_PREPROCESSED_PATH, output_text_file=config.CC100_CORRUPTED_PATH):
        if input_text_file is None:
            input_text_file=config.CC100_PREPROCESSED_PATH
        if output_text_file is None:
            output_text_file = config.CC100_CORRUPTED_PATH
        self.session = Session(input_text_file, output_text_file, 'corrupted')
    
    def run(self):
        self.session.run()

class Shrink:
    def __init__(self, input_text_file, output_text_file) -> None:
        self.input_text_file = input_text_file
        self.output_text_file = output_text_file

    def run(self):
        command = f'head -{config.CC100_SMALL_SIZE} {self.input_text_file} >> {self.output_text_file}'
        os.system(command)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input text file')
    parser.add_argument('--output_file', type=str, help='output text file')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    Preprocess(input_file, output_file).run()
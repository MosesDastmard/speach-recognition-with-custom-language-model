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
from .loader import Loader
if config.MODE == 'small':
    bpe_tokenizer = BPE.load(config.TOKENIZER_MODEL_SMALL_PATH)
else:
    bpe_tokenizer = BPE.load(config.TOKENIZER_MODEL_PATH)


def join_tokens(corrupted_sentence, clean_sentence):
    tokens_clean = bpe_tokenizer.tokenize(clean_sentence, result='tokens')
    tokens_corrupted = bpe_tokenizer.tokenize(corrupted_sentence, result='tokens')
    if len(tokens_clean) >= config.MAX_TOKEN_INPUT:
        return []
    if len(tokens_corrupted) >= config.MAX_TOKEN_INPUT:
        return []
    joint_tokens = "~".join(tokens_corrupted) + "|" + "~".join(tokens_clean)
    return [joint_tokens]

def generate_line(clean_sentence):
    corrupted_sentence = mapper(clean_sentence)
    return join_tokens(corrupted_sentence, clean_sentence)


def slice_sentence(sentence):
    window = int(config.MAX_TOKEN_INPUT*0.7)
    stride = window//2
    words = nltk.word_tokenize(sentence, language='italian')
    sentences = []
    if len(words) < window:
        sentences.append(sentence)
    else:
        start = 0
        end = window
        while end < len(words):
            sentence_ = " ".join(words[start:end])
            sentences.append(sentence_)
            start += stride
            end += stride
        if start < len(words):
            sentence_ = " ".join(words[start:])
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
            ref_rdd = ref_rdd.flatMap(generate_line)
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


class Validation:
    def __init__(self, validation_path):
        self.validation_path = validation_path
        self.loader = Loader()
    def read(self):
        self.clean_sentences = self.loader.get_actual_sentences()
        self.corrupted_sentences = self.loader.get_w2v_predictions()
    
    def create(self):
        for corrupted, clean in zip(self.clean_sentences, self.corrupted_sentences):
            line = join_tokens(corrupted_sentence=corrupted, clean_sentence=clean)
            if len(line) > 0:
                with open(self.validation_path, 'a') as f:
                    f.write(line[0] + "\n")

    def run(self):
        self.read()
        self.create()


class SimpleBiLSTMCharLevel:
    class Train:
        def __init__(self) -> None:
            self.clean_text_path = config.CC100_PREPROCESSED_PATH
            self.output_path = config.SimpleBiLSTMCharLevel_train_path
        
        def start_saprk_session(self):
            cores_number = multiprocessing.cpu_count()
            self.spark = SparkSession.builder.appName('dataset_builder').master(f'local[{cores_number}]').getOrCreate()
            print(f"spark runs on {cores_number} cores.")
        
        def run(self):
            self.start_saprk_session()
            

def string_list_to_text(string_list, file_path):
    with open(file_path, 'w') as f:
        string_list = [line + "\n" for line in string_list]
        f.writelines(string_list)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input text file')
    parser.add_argument('--output_file', type=str, help='output text file')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    Preprocess(input_file, output_file).run()
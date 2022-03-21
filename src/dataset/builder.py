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


class Corrupted:
    def __init__(self, input_text_file=config.CC100_PREPROCESSED_PATH, output_text_file=config.CC100_CORRUPTED_PATH):
        if input_text_file is None:
            input_text_file=config.CC100_PREPROCESSED_PATH
        if output_text_file is None:
            output_text_file = config.CC100_CORRUPTED_PATH
        self.session = Session(input_text_file, output_text_file, 'corrupted')
    
    def run(self):
        self.session.run()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input text file')
    parser.add_argument('--output_file', type=str, help='output text file')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    Preprocess(input_file, output_file).run()
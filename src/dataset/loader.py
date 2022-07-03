import os
from src.util import config
import pandas as pd
from libs.datil.flag import Flag
from numpy import loadtxt
import numpy as np
from src.util.functions import purify_text
from tqdm import tqdm
from jiwer import wer

class Loader:
    def __init__(self):
        test_path = os.path.join(config.COMMON_VOICE_DATA_PATH, 'test.tsv')
        self.test_df = pd.read_csv(test_path, sep='\t')
        train_path = os.path.join(config.COMMON_VOICE_DATA_PATH, 'train.tsv')
        self.train_df = pd.read_csv(train_path, sep='\t')
        self.flag = Flag('w2v')
    
    @staticmethod
    def get_clip_path(clip_name):
        return os.path.join(config.CLIPS_PATH, clip_name)
    
    def is_test(self, clip_name):
        if clip_name in self.test_df['path'].values:
            return True
        if clip_name in self.train_df['path'].values:
            return False
        raise Exception(f'{clip_name} does not exist.')
        
    def exists(self, clip_name):
        clip_path = self.get_clip_path(clip_name)
        return self.flag.exists(clip_path)
    
    def get_(self, clip_name, func):
        if self.exists(clip_name):
            return func(clip_name)
        else:
            raise Exception(f'array for {clip_name} does not exist')
            
    @staticmethod
    def read_array(clip_name):
        array_path = os.path.join(config.ARRAYS_PATH, clip_name + config.ARRAY_EXTENTION)
        a = np.fromfile(array_path)
        return a
    
    def get_array(self, clip_name):
        return self.get_(clip_name, self.read_array)
        
    @staticmethod
    def read_logit(clip_name):
        logit_path = os.path.join(config.LOGITS_PATH, clip_name + config.LOGITS_EXTENTION)
        a = loadtxt(logit_path, delimiter=",")
        a = a[:, [0] + list(range(4,44))]
        return a
    
    def get_logit(self, clip_name):
        return self.get_(clip_name, self.read_logit)
    
    @staticmethod
    def read_prediction_w2v(clip_name):
        prediction_path = os.path.join(config.W2V_PREDICTION_PATH, clip_name + config.PREDICTION_EXTENTION)
        with open(prediction_path, 'r') as file:
            line = file.readline()
        return line
    
    def get_w2v_prediction(self, clip_name):
        return self.get_(clip_name, self.read_prediction_w2v)
    
    def get_w2v_predictions(self):
        clip_names = self.get_test()
        predictions = []
        for clip_name in clip_names:
            predicted_sentence = self.get_w2v_prediction(clip_name)
            predictions.append(predicted_sentence)
        return predictions

    def get_actual_sentence(self, clip_name):
        if self.is_test(clip_name):
            text = self.test_df[self.test_df['path'] == clip_name]['sentence'].values[0]
            text = purify_text(text)
            return text
        else:
            text = self.train_df[self.train_df['path'] == clip_name]['sentence'].values[0]
            text = purify_text(text)
            return text

    def get_actual_sentences(self, mode='test'):
        if mode == 'test':
            clip_names = self.get_test()
        elif mode == 'train':
            clip_names = self.get_train()
        actuals = []
        for clip_name in tqdm(clip_names):
            actual_sentence = self.get_actual_sentence(clip_name)
            actuals.append(actual_sentence)
        return actuals

    def get_test(self):
        return self.test_df['path'].tolist()
    
    def get_train(self):
        return self.train_df['path'].tolist()

    def get_train_actual_senteneces(self):
        return self.train_df['sentence'].apply(purify_text).values.tolist()


    def get_WER_W2V(self):
        actual_sentences, predicted_sentences_w2v = self.get_W2V_evaluation_data()
        return wer(actual_sentences, predicted_sentences_w2v)

    def get_W2V_evaluation_data(self):
        actual_sentences_ = self.get_actual_sentences()
        predicted_sentences_w2v_ = self.get_w2v_predictions()

        if len(actual_sentences_) != len(predicted_sentences_w2v_):
            raise Exception("number of object in grandtruth and predicted is not the same")
        
        actual_sentences = []
        predicted_sentences_w2v = []

        for act, pre in zip(actual_sentences_, predicted_sentences_w2v_):
            if len(act) > 4 and len(pre) > 4:
                actual_sentences.append(act)
                predicted_sentences_w2v.append(pre)
        return actual_sentences, predicted_sentences_w2v



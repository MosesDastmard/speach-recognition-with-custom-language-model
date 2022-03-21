import librosa
import pandas as pd
from src.util import config
import numpy as np
from numpy import asarray
from numpy import savetxt
import os
from asrecognition import ASREngine
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
from libs.datil.flag import Flag
from tqdm import tqdm

asr = ASREngine(config.MODEL_LANGUAGE, model_path=config.MODEL_W2V_PATH)
processor = Wav2Vec2Processor.from_pretrained(config.MODEL_W2V_PATH)
model = Wav2Vec2ForCTC.from_pretrained(config.MODEL_W2V_PATH)

def audio_to_array(clip_path, array_path, logits_path, sentence_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sr = librosa.load(clip_path, sr=config.SAMPLE_RATE)
        speech_array.tofile(array_path)
        inputs = processor(speech_array, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0]
        with open(sentence_path, 'w') as f:
            f.write(predicted_sentence)
        logits = logits[0,:]
        a = asarray(logits.numpy(), dtype=np.float64)
        savetxt(logits_path, a, delimiter=',')


class W2V:
    def __init__(self):
        os.makedirs(config.CLIPS_PATH, exist_ok=True)
        os.makedirs(config.ARRAYS_PATH, exist_ok=True)
        os.makedirs(config.LOGITS_PATH, exist_ok=True)
        os.makedirs(config.W2V_PREDICTION_PATH, exist_ok=True)

    def get_clip_files(self):
        clip_list = pd.read_csv(os.path.join(config.COMMON_VOICE_DATA_PATH, 'test.tsv'), sep='\t')
        return clip_list['path']

    def run(self):
        clip_list = self.get_clip_files()
        flag = Flag('w2v')
        for clip_name in tqdm(clip_list):
            if clip_name.lower().endswith(".mp3"):
                clip_path = os.path.join(config.CLIPS_PATH, clip_name)
                if not flag.exists(clip_path):
                    array_path = os.path.join(config.ARRAYS_PATH, clip_name + config.ARRAY_EXTENTION)
                    prediction_path = os.path.join(config.W2V_PREDICTION_PATH, clip_name + config.PREDICTION_EXTENTION)
                    logit_path = os.path.join(config.LOGITS_PATH, clip_name + config.LOGITS_EXTENTION)
                    audio_to_array(clip_path, array_path, logit_path, prediction_path)
                    flag.put(clip_path)

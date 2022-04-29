import os
DATA_PATH = 'data/'
ERRORS_PATH = DATA_PATH + "errors/"
NULL_CHAR = "?"
IGNORE_PUNC = ","
CC100_PREPROCESSED_PATH = DATA_PATH + "it.preprocessed.txt"
CC100_PREPROCESSED_SMALL_PATH = DATA_PATH + "it.preprocessed.small.txt"
CC100_SMALL_SIZE = 400000
CC100_CORRUPTED_PATH = DATA_PATH + "it.corrupted.txt"
CC100_CORRUPTED_SMALL_PATH = DATA_PATH + "it.corrupted.small.txt"
CC100_CLEAN_CORRUPTED_PATH = DATA_PATH + "it.clean.corrupted.txt"
CC100_CLEAN_CORRUPTED_SMALL_PATH = DATA_PATH + "it.clean.corrupted.small.txt"
MAX_TOKEN_INPUT = 32
CHAR_SET = [
            ' ',
            "'",
            '-',
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'G',
            'H',
            'I',
            'J',
            'K',
            'L',
            'M',
            'N',
            'O',
            'P',
            'Q',
            'R',
            'S',
            'T',
            'U',
            'V',
            'W',
            'X',
            'Y',
            'Z',
            'À',
            'Á',
            'È',
            'É',
            'Ì',
            'Í',
            'Ò',
            'Ó',
            'Ù',
            'Ú',
            'Š',
]
CHAR_SET = [c.lower() for c in CHAR_SET]
W2V_CHAR_SET = [
                '',
                ' ',
                "'",
                '-',
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z',
                'à',
                'á',
                'è',
                'é',
                'ì',
                'í',
                'ò',
                'ó',
                'ù',
                'ú',
                'š'
                ]
MAX_CHAR = 100
MAX_TOKEN_LEN = MAX_CHAR + 2
MAX_CHUNK_SIZE = 2048
STRIDE = 25
PARTITION_SIZE = 16000000 #16/100Mbyte
VOCAB_SIZE = 40000
###################### DATASET ###########################
COMMON_VOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-7.0-2021-07-21/cv-corpus-7.0-2021-07-21-it.tar.gz"
CC100_URL = "http://data.statmt.org/cc-100/it.txt.xz"
CC100_TAR_PATH = DATA_PATH + "it.txt.xz"
COMMON_VOICE_TAR_PATH = DATA_PATH + "cv-corpus-7.0-2021-07-21-it.tar.gz"
COMMON_VOICE_PATH = DATA_PATH + "cv-corpus-7.0-2021-07-21-it"
CC100_PATH = DATA_PATH + "it.txt"
CC100_SMALL_PATH = DATA_PATH + "it.small.txt"
MODEL_DIR = "model/"
TOKENIZER_MODEL_PATH = MODEL_DIR + "tokenizer.json"
TOKENIZER_MODEL_SMALL_PATH = MODEL_DIR + "tokenizer.small.json"
KENLM_MODEL_PATH = MODEL_DIR + "it.arpa"
MAPPING_MODEL_PATH = MODEL_DIR + "mapping.json"
KENLM_TMP_DIR = MODEL_DIR + "tmp"
ARRAY_EXTENTION = ".array"
SAMPLE_RATE = 16000
MODEL_LANGUAGE = "it"
MODEL_W2V_PATH = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
PREDICTION_EXTENTION = ".txt"
LOGITS_EXTENTION = '.logits'
COMMON_VOICE_DATA_PATH = os.path.join(COMMON_VOICE_PATH ,"cv-corpus-7.0-2021-07-21", 'it')
CLIPS_PATH = os.path.join(COMMON_VOICE_DATA_PATH, 'clips')
ARRAYS_PATH = os.path.join(COMMON_VOICE_DATA_PATH, 'array')
LOGITS_PATH = os.path.join(COMMON_VOICE_DATA_PATH, 'logits')
W2V_PREDICTION_PATH = os.path.join(COMMON_VOICE_DATA_PATH, 'w2v_predictions')
MODE = 'full'
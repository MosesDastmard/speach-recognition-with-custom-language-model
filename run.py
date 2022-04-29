# from src.dataset.download import Download, Extract
# from src.kenlm import TrainKenlm
from src.util import config
from libs.datil.flag import Flag
# from src.prediction.w2v import W2V
from src.error import Error
from src.tokenizer.bpe import BPE

# #download common voice dataset
# flag = Flag('commonvoice')
# if not flag.exists(config.COMMON_VOICE_PATH):
#     Download(config.COMMON_VOICE_URL, config.COMMON_VOICE_TAR_PATH).run()
#     Extract(config.COMMON_VOICE_TAR_PATH, config.COMMON_VOICE_PATH).run()
#     flag.put(config.COMMON_VOICE_PATH)



# # download text dataset
# flag = Flag('CC100')
# if not flag.exists(config.CC100_PATH):
#     Download(config.CC100_URL, config.CC100_TAR_PATH).run()
#     Extract(config.CC100_TAR_PATH, config.CC100_PATH).run()
#     flag.put(config.CC100_PATH)


# flag = Flag('preprocessing')
# if config.MODE == 'small':
#     output_file = config.CC100_PREPROCESSED_SMALL_PATH
#     input_file = config.CC100_SMALL_PATH
#     flag_shrink = Flag('shrink')
#     if not flag_shrink.exists(input_file):
#         Shrink(config.CC100_PATH, config.CC100_SMALL_PATH).run()
#         flag_shrink.put(input_file)
# if not flag.exists(output_file):
#     Preprocess(input_file, output_file).run()
#     flag.put(output_file)

# flag = Flag('trainkenlm')
# if not flag.exists(config.KENLM_MODEL_PATH):
#     TrainKenlm(config.KENLM_MODEL_PATH, config.CC100_PREPROCESSED_PATH).run()
#     flag.put(config.KENLM_MODEL_PATH)

# flag = Flag('w2v')
# if not flag.exists(config.COMMON_VOICE_DATA_PATH):
#     W2V().run()
#     flag.put(config.COMMON_VOICE_DATA_PATH)

# flag = Flag('error')
# if not flag.exists(config.MAPPING_MODEL_FILE):
#     Error.run()
#     flag.put(config.MAPPING_MODEL_FILE)


# flag = Flag('corrupted')
# if config.MODE == 'small':
#     input_file = config.CC100_PREPROCESSED_SMALL_PATH
#     output_file = config.CC100_CORRUPTED_SMALL_PATH
# else:
#     input_file = config.CC100_PREPROCESSED_PATH
#     output_file = config.CC100_CORRUPTED_PATH

# if not flag.exists(output_file):
#     Corrupted(input_file, output_file).run()
#     flag.put(output_file)




# flag = Flag('bpe')
# if config.MODE == 'small':
#     input_files = [config.CC100_PREPROCESSED_SMALL_PATH, config.CC100_CORRUPTED_SMALL_PATH]
#     model_path = config.TOKENIZER_MODEL_SMALL_PATH
# else:
#     input_files = [config.CC100_PREPROCESSED_PATH, config.CC100_CORRUPTED_PATH]
#     model_path = config.TOKENIZER_MODEL_PATH

# if not flag.exists(model_path):
#     BPE(input_files, model_path).train()
#     flag.put(model_path)

from src.dataset.builder import Preprocess, Corrupted, CleanCorrupted, Shrink

flag = Flag('clean.corrupted')
if config.MODE == 'small':
    input_file = config.CC100_PREPROCESSED_SMALL_PATH
    output_file = config.CC100_CLEAN_CORRUPTED_SMALL_PATH
else:
    input_file = config.CC100_PREPROCESSED_PATH
    output_file = config.CC100_CLEAN_CORRUPTED_PATH
if not flag.exists(output_file):
    CleanCorrupted(input_file, output_file).run()
    flag.put(output_file)

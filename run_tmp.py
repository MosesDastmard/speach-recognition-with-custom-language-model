from telnetlib import DO
from src.dataset.download import Download, Extract
from src.kenlm import TrainKenlm
from src.dataset.builder import Preprocess
from src.util import config
from libs.datil.flag import Flag
from src.prediction.w2v import W2V
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
# if not flag.exists(config.CC100_PREPROCESSED_PATH):
#     Preprocess(config.CC100_PATH, config.CC100_PREPROCESSED_PATH).run()
#     flag.put(config.CC100_PREPROCESSED_PATH)

flag = Flag('trainkenlm')
if not flag.exists(config.KENLM_MODEL_PATH):
    TrainKenlm(config.KENLM_MODEL_PATH, config.CC100_PREPROCESSED_PATH).run()
    flag.put(config.KENLM_MODEL_PATH)

# flag = Flag('w2v')
# if not flag.exists(config.COMMON_VOICE_DATA_PATH):
#     W2V().run()
#     flag.put(config.COMMON_VOICE_DATA_PATH)
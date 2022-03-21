import os
from src.util import config
from libs.datil.flag import Flag
from src.util.functions import Process

class TrainKenlm:
    def __init__(self, model_path, file_path) -> None:
        self.model_path = model_path
        self.file_path = file_path

    def run(self):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        train_command = f"./libs/kenlm/build/bin/lmplz --arpa={self.model_path} --text={self.file_path} -o 3 -T {config.KENLM_TMP_DIR} -S 50% --skip_symbols"
        # Process(train_command)
        os.system(train_command)

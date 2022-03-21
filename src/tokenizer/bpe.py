from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE as BPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from src.util import config

class BPE:
    def __init__(self, input_files, model_path) -> None:
        self.input_files = input_files
        self.model_path = model_path

    def train(self):
        tokenizer = Tokenizer(BPETokenizer())
        trainer = BpeTrainer(vocab_size=4000)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train(self.input_files, trainer)
        tokenizer.save(self.model_path)
    
    @staticmethod
    def load(model_path):
        return Tokenizer.from_file(model_path)
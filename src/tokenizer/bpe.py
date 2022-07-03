from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE as BPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from src.util import config
import tensorflow as tf

class BPE:
    def __init__(self, input_files, model_path) -> None:
        self.input_files = input_files
        self.model_path = model_path

    def train(self, vocab_size, end_of_word_suffix):
        tokenizer = Tokenizer(BPETokenizer(unk_token='[UNK]'))
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['[UNK]', '[START]', '[END]', '[PAD]'], end_of_word_suffix=end_of_word_suffix, )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train(files=self.input_files, trainer=trainer)
        tokenizer.save(self.model_path)
    
    @staticmethod
    def load(model_path):
        base_tokenizer = Tokenizer.from_file(model_path)
        class tokenizer:
            def __init__(self):
                self.start_token = '[START]'
                self.end_token = '[END]'
                self.start_id = self.__get_start_id()
                self.end_id = self.__get_end_id()

            def encode(self, input):
                return base_tokenizer.encode(input)

            def decode(self, input):
                return base_tokenizer.decode(input)

            def tokenize(self, input, result='ids'):
                if result == 'ids':
                    return [self.start_id] + base_tokenizer.encode(input).ids + [self.end_id]
                if result == 'tokens':
                    return [self.start_token] + base_tokenizer.encode(input).tokens + [self.end_token]

            def detokenize(self, ids):
                joint_tokens = base_tokenizer.decode(ids)
                return joint_tokens.replace(" ", "").replace("_", " ").strip()
            
            def __get_start_id(self):
                return base_tokenizer.encode(self.start_token).ids[0]
            
            def __get_end_id(self):
                return base_tokenizer.encode(self.end_token).ids[0]

            def get_tf_lookup_table(self):
                vocabs = base_tokenizer.get_vocab()
                keys = tf.constant(list(vocabs.keys()))
                values = tf.constant(list(vocabs.values()), dtype=tf.int64)
                init = tf.lookup.KeyValueTensorInitializer(keys=keys,values=values)
                table = tf.lookup.StaticVocabularyTable(init,num_oov_buckets=5)
                return table

            def get_vocab_size(self):
                return base_tokenizer.get_vocab_size()
            
            def get_vocab(self):
                return base_tokenizer.get_vocab()

        return tokenizer()



class Simple:
    def __init__(self) -> None:
        pass
    def train(self, token_set):
        tokens = {'[START]':1,
          '[END]':2,
          '[PAD]':3,
          '[UNK]':4}
        for i,token in enumerate(token_set):
            tokens[token] = i + 5
        return tokens

    @staticmethod
    def load(tokens_set):

        class tokenizer:
            def __init__(self):
                self.start_token = '[START]'
                self.end_token = '[END]'
                self.start_id = self.__get_start_id()
                self.end_id = self.__get_end_id()
                self.tokens = Simple().train(tokens_set)
                self.reverse_tokens = {i:c for c,i in self.tokens.items()}

            def tokenize(self, input, result='ids'):
                if result == 'ids':
                    return [self.start_id] + [self.tokens[i] for i in input if i in self.tokens.keys()] + [self.end_id]
                if result == 'tokens':
                    return [self.start_token] + list(input) + [self.end_token]

            def detokenize(self, ids):
                joint_tokens = [self.reverse_tokens[id] for id in ids if id in self.reverse_tokens.keys()]
                return joint_tokens
            
            def __get_start_id(self):
                return 1
            
            def __get_end_id(self):
                return 2

            def get_vocab_size(self):
                return len(self.tokens) + 4

            def get_tf_lookup_table(self):
                vocabs = self.tokens
                keys = tf.constant(list(vocabs.keys()))
                values = tf.constant(list(vocabs.values()), dtype=tf.int64)
                init = tf.lookup.KeyValueTensorInitializer(keys=keys,values=values)
                table = tf.lookup.StaticVocabularyTable(init,num_oov_buckets=1)
                return table
        return tokenizer()
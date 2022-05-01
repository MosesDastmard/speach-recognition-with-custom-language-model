import tensorflow as tf
from src.util import config
import os
import logging
from jiwer import wer
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    logging.disable(logging.WARNING)
except AttributeError:
    pass

import numpy as np

from contextlib import redirect_stdout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar, GeneratorEnqueuer
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, name="dec_embedding")
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, name=f"dec_layer_{i}") for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, name="dec_dropout")

        self.dec_output = tf.keras.layers.Dense(target_vocab_size, name="dec_dense")

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, look_ahead_mask)


        # x.shape == (batch_size, target_seq_len, d_model)
        output = self.dec_output(x)

        return output

    def get_config(self):
        """Return the config of the layer"""

        config = super(Decoder, self).get_config()
        return config


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="dec_layer"):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention_1")

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_2")
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_3")

        self.dropout1 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_2")
        self.dropout3 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_3")

    def call(self, x, look_ahead_mask=None):

        attn, _ = self.mha(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn = self.dropout1(attn)
        out = self.layernorm1(attn + x)


        ffn_output = self.ffn(out)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out = self.layernorm3(ffn_output)  # (batch_size, target_seq_len, d_model)

        return out

    def get_config(self):
        """Return the config of the layer"""

        config = super(DecoderLayer, self).get_config()
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def get_config(self):
        """Return the config of the layer"""

        config = super(MultiHeadAttention, self).get_config()
        return config


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks(inp=None, tar=None):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


@tf.function
def loss_func(y_true, y_pred):
    label_length = tf.ones_like(y_true)[:,0]*MAX_TOKENS
    # logit_length = tf.constant(y_pred.shape[1], dtype=tf.int32, shape=y_pred.shape[0])
    # print(y_true.shape)
    # print(y_pred.shape)
    # print(label_length.shape)
    # print(logit_length.shape)
    # exit()
    losses = tf.nn.ctc_loss(
                            labels=y_true,
                            logits=y_pred,
                            label_length=label_length,
                            logit_length=label_length,
                            logits_time_major=False,
                            blank_index=0,
                            )
    # losses = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return tf.reduce_mean(losses)


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    losses = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(losses)
    print(loss.shape)
    return loss


def dummy_loss_func(y_true, y_pred):
    return tf.reduce_sum(tf.constant([1]))

num_layers=6
units=512
d_model=512
num_heads=8
dropout=0.1
vocab_size=config.VOCAB_SIZE+10
initial_step=4000

dec_input = Input(shape=(None,), name="dec_input")
# enc_shift = Input(shape=(None,), name="enc_shift")
enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(dec_input, dec_input)

decoder = Decoder(num_layers=num_layers,
                       d_model=d_model,
                       num_heads=num_heads,
                       dff=units,
                       target_vocab_size=vocab_size,
                       maximum_position_encoding=vocab_size,
                       rate=dropout)

dec_output = decoder(dec_input, dec_input, look_ahead_mask, dec_padding_mask)

learning_rate = CustomSchedule(d_model=d_model, initial_step=initial_step)


optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def logit_to_pred(logit):
    pred = tf.argmax(logit, axis=2)
    return pred

def accuracy_logit(real, logit):
    pred = logit_to_pred(logit)
    return accuracy_pred(real, pred)

def accuracy_pred(real, pred):
    pred = tf.pad(pred, [[0,0],[0,1]])
    real = tf.pad(real, [[0,0],[0,1]])
    # return pred
    real_end_mask = tf.cast(tf.equal(real, 2), dtype=tf.int64)
    real_end_mask = tf.roll(real_end_mask, 1, axis=1)
    real_end_mask = tf.math.cumsum(real_end_mask, axis=1)
    real_end_mask = tf.cast(tf.equal(real_end_mask, 0), dtype=tf.int64)
    pred_end_mask = tf.cast(tf.equal(pred, 2), dtype=tf.int64)
    pred_end_mask = tf.roll(pred_end_mask, 1, axis=1)
    pred_end_mask = tf.math.cumsum(pred_end_mask, axis=1)
    pred_end_mask = tf.cast(tf.equal(pred_end_mask, 0), dtype=tf.int64)
    # return pred_end_mask
    accuracies = tf.equal(tf.cast(real, tf.int64), tf.cast(pred, tf.int64))
    mask = tf.not_equal(real_end_mask + pred_end_mask, 0)
    # mask = tf.not_equal(real_end_mask, 0)
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # return mask
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)*100

def wer_pred(real, pred):
    pred = tf.cast(pred, dtype=tf.int64)
    real = tf.cast(real, dtype=tf.int64)
    pred = tf.pad(pred, [[0,0],[0,1]])
    real = tf.pad(real, [[0,0],[0,1]])
    # return pred
    real_end_mask = tf.cast(tf.equal(real, 2), dtype=tf.int64)
    real_end_mask = tf.roll(real_end_mask, 1, axis=1)
    real_end_mask = tf.math.cumsum(real_end_mask, axis=1)
    real_end_mask = tf.cast(tf.equal(real_end_mask, 0), dtype=tf.int64)
    pred_end_mask = tf.cast(tf.equal(pred, 2), dtype=tf.int64)
    pred_end_mask = tf.roll(pred_end_mask, 1, axis=1)
    pred_end_mask = tf.math.cumsum(pred_end_mask, axis=1)
    pred_end_mask = tf.cast(tf.equal(pred_end_mask, 0), dtype=tf.int64)
    pred *= pred_end_mask
    real *= real_end_mask
    real_list = real.numpy().tolist()
    pred_list = pred.numpy().tolist()
    real_sentences = []
    pred_sentences = []
    for real_, pred_ in zip(real_list, pred_list):
        real_sentence = bpe_tokenizer.detokenize(real_)
        real_sentences.append(real_sentence)
        pred_sentence = bpe_tokenizer.detokenize(pred_)
        pred_sentences.append(pred_sentence)
    return wer(real_sentences, pred_sentences)

def logit_greedy_decoder(logit):
    ctc_logits = tf.transpose(logit, perm=[1,0,2])
    print(ctc_logits.shape)
    logit_length = tf.cast(tf.ones_like(logit)[:,0,0]*MAX_TOKENS, dtype=tf.int32)
    # print(logit_length); exit()
    decoded_output, _ = tf.nn.ctc_greedy_decoder(
                ctc_logits, 
                logit_length, 
                merge_repeated=True, 
                blank_index=0,
            )
    output = decoded_output
    print(output.shape); exit()
    return 

def wer_logit(real, logit):
    pred = logit_to_pred(logit)
    # print(pred.shape)
    # print(real.shape)
    # exit()
    return wer_pred(real, pred)


# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
model = Model(inputs=dec_input, outputs=dec_output, name="transformer")
# WER compiler
# model.compile(optimizer=optimizer, 
#               loss=[loss_func, dummy_loss_func], 
#               loss_weights=[1, 0], 
#               metrics=[[accuracy_logit, wer_logit], [accuracy_pred, wer_pred]],
#               run_eagerly=True)

# Fast compiler
model.compile(optimizer=optimizer, 
              loss=loss_func, 
              metrics=[accuracy_logit],
              run_eagerly=True)

from src.tokenizer.bpe import BPE
from src.util import config
# if config.MODE == 'small':
#     tokenizer_path = config.TOKENIZER_MODEL_SMALL_PATH
# else:
tokenizer_path = config.TOKENIZER_MODEL_PATH

bpe_tokenizer = BPE.load(tokenizer_path)
lookup_table = bpe_tokenizer.get_tf_lookup_table()
BUFFER_SIZE = 20000
BATCH_SIZE = 16


if config.MODE == 'small':
    data_path = config.CC100_CLEAN_CORRUPTED_SMALL_PATH
else:
    data_path = config.CC100_CLEAN_CORRUPTED_PATH

file_paths = []
for file in os.listdir(data_path):
    if file.startswith('part'):
        file_paths.append(os.path.join(data_path, file))
print(file_paths)



MAX_TOKENS = config.MAX_TOKEN_INPUT + 2

def to_ids(x):
    l = tf.strings.split(x, '|')
    return tf.RaggedTensor.from_tensor(tf.expand_dims(lookup_table[tf.strings.split(l[0], "~")],1)), tf.RaggedTensor.from_tensor(tf.expand_dims(lookup_table[tf.strings.split(l[1], '~')],1))

def make_batches(ds):
    return (ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))

def reshape(error, clean):
    error = error[:,:,0].to_tensor(shape=(None, MAX_TOKENS+1))
    error_ids = error[:,1:]
    clean = clean[:,:,0].to_tensor(shape=(None, MAX_TOKENS+1))
    output_ids = clean[:,1:]
    return error_ids, output_ids

def get_train_batches(file_path):
    train_examples = tf.data.TextLineDataset(file_path)
    train_examples = train_examples.map(to_ids)
    train_batches = make_batches(train_examples)
    train_batches = train_batches.map(reshape)
    return train_batches

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ckpt_manager.save()
        # print('checkpoint created on epoch end')
# train_suffix = 'test1small' # --> 256 d_model
train_suffix = 'CTCtestv2' # --> 512 d_odel
checkpoint_path = f'./checkpoints/{train_suffix}/train'

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

from libs.datil.flag import Flag

if __name__ == "__main__":
    for epoch in range(50):
        print(f"epoch:{epoch}")
        flag = Flag(f'epoch_{epoch}_{train_suffix}')
        for file_path in file_paths:
            if not flag.exists(file_path):
                print(file_path)
                train_batches = get_train_batches(file_path)
                # for (error, error_shifted), clean in train_batches:
                #     print(accuracy_pred(clean, error))
                model.fit(train_batches, epochs=1, callbacks=[CustomCallback()])
                flag.put(file_path)



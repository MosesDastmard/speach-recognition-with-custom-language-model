import tensorflow as tf

import os
import logging

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



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, name="enc_embedding")
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, f"enc_layer_{i}") for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, name="enc_dropout")

    def call(self, x, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        """Return the config of the layer"""

        config = super(Encoder, self).get_config()
        return config


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="enc_layer"):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention")
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_2")

        self.dropout1 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_2")

    def call(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):
        """Return the config of the layer"""

        config = super(EncoderLayer, self).get_config()
        return config


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
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        output = self.dec_output(x)

        return output, attention_weights

    def get_config(self):
        """Return the config of the layer"""

        config = super(Decoder, self).get_config()
        return config


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="dec_layer"):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention_1")
        self.mha2 = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention_2")

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_2")
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_3")

        self.dropout1 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_2")
        self.dropout3 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_3")

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

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


def loss_func(y_true, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

num_layers=6
units=512
d_model=256
num_heads=8
dropout=0.1
stop_tolerance=20
reduce_tolerance=15
vocab_size=210
initial_step=4000

enc_input = Input(shape=(None,), name="enc_input")
dec_input = Input(shape=(None,), name="dec_input")
enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(enc_input, dec_input)

encoder = Encoder(num_layers=num_layers,
                       d_model=d_model,
                       num_heads=num_heads,
                       dff=units,
                       input_vocab_size=vocab_size,
                       maximum_position_encoding=vocab_size,
                       rate=dropout)

decoder = Decoder(num_layers=num_layers,
                       d_model=d_model,
                       num_heads=num_heads,
                       dff=units,
                       target_vocab_size=vocab_size,
                       maximum_position_encoding=vocab_size,
                       rate=dropout)

enc_output = encoder(enc_input, enc_padding_mask)
dec_output, _ = decoder(dec_input, enc_output, look_ahead_mask, dec_padding_mask)

learning_rate = CustomSchedule(d_model=d_model, initial_step=initial_step)


optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy_function(real, pred):
    accuracies = tf.equal(tf.cast(real, tf.int64), tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
model = Model(inputs=[enc_input, dec_input], outputs=dec_output, name="transformer")
model.compile(optimizer=optimizer, loss=loss_func, metrics=[accuracy_function])

from src.tokenizer.bpe import BPE
from src.util import config
bpe_tokenizer = BPE.load(config.TOKENIZER_MODEL_PATH)
lookup_table = bpe_tokenizer.get_tf_lookup_table()
BUFFER_SIZE = 20000
BATCH_SIZE = 128



data_path = os.path.join(config.CC100_CLEAN_CORRUPTED_PATH)
file_paths = []
for file in os.listdir(data_path):
    if file.startswith('part'):
        file_paths.append(os.path.join(data_path, file))
file_paths



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
    clean_ids = clean[:,:-1]
    output_ids = clean[:,1:]
    return (error_ids, clean_ids), output_ids

def get_train_batches(file_path):
    train_examples = tf.data.TextLineDataset(file_path)
    train_examples = train_examples.map(to_ids)
    train_batches = make_batches(train_examples)
    train_batches = train_batches.map(reshape)
    return train_batches

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ckpt_manager.save()
        print('checkpoint created on epoch end')

checkpoint_path = './checkpoints/transformerv9/train'

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

from libs.datil.flag import Flag

if __name__ == "__main__":
    for epoch in range(20):
        print(f"epoch:{epoch}")
        flag = Flag(f'epoch_{epoch}')
        for file_path in file_paths:
            if not flag.exists(file_path):
                print(file_path)
                train_batches = get_train_batches(file_path)
                model.fit(train_batches, epochs=1, callbacks=[CustomCallback()])
                flag.put(file_path)



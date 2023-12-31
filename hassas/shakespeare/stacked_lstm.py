import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, types='H', masks=None, optimizer=None):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.type = types
        self.masks = masks
        super(ClientModel, self).__init__(seed, lr, optimizer)

    def create_model(self):
        if self.type == 'H':
            features = tf.placeholder(tf.int32, [None, self.seq_len])
            embedding = tf.get_variable("embedding", [self.num_classes, 8], trainable=False)
            x = tf.nn.embedding_lookup(embedding, features)
            labels = tf.placeholder(tf.int32, [None, self.num_classes])
            
            stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
            outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)

            pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            eval_metric_ops = tf.count_nonzero(correct_pred)

            return features, labels, train_op, eval_metric_ops, loss, None

        elif self.type == 'L':
            features = tf.placeholder(tf.int32, [None, self.seq_len])
            embedding = tf.get_variable("embedding", [self.num_classes, 8], trainable=False)
            x = tf.nn.embedding_lookup(embedding, features)
            labels = tf.placeholder(tf.int32, [None, self.num_classes])
            
            mask = self.masks['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0'][2][1]
            n_hidden = int(np.count_nonzero(mask)/4)
            stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(2)])
            outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)

            pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            eval_metric_ops = tf.count_nonzero(correct_pred)

            return features, labels, train_op, eval_metric_ops, loss, None

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch

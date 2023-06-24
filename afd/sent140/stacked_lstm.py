import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn
import json

from model import Model
from utils.language_utils import line_to_indices, get_word_emb_arr, val_to_vec


VOCAB_DIR = 'sent140/embs.json'
with open(VOCAB_DIR, 'r') as inf:
    embs = json.load(inf)
id2word = embs['vocab']
word2id = {v: k for k,v in enumerate(id2word)}
word_emb = np.array(embs['emba'])

class ClientModel(Model):

    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, types='H', masks=None, optimizer=None, emb_arr=None):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.vocab_size = len(vocab)
        self.emb_arr = word_emb

        self.type = types
        self.masks = masks
        super(ClientModel, self).__init__(seed, lr, optimizer)

    def create_model(self):
        if self.type == 'H':
            features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
            labels = tf.placeholder(tf.int64, [None,], name='labels')
            # labels = tf.placeholder(tf.int64, [None, self.num_classes])

            embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
            x = tf.nn.embedding_lookup(embs, features)
            
            stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
            outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
            fc1 = tf.layers.dense(inputs=outputs[:,-1,:], units=30, name='dense1')
            pred = tf.squeeze(tf.layers.dense(inputs=fc1, units=1))
            
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pred)
            grads_and_vars = self.optimizer.compute_gradients(loss)
            grads, _ = zip(*grads_and_vars)
            train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
            correct_pred = tf.equal(tf.to_int64(tf.greater(pred,0)), labels)
            eval_metric_ops = tf.count_nonzero(correct_pred)
            
            return features, labels, train_op, eval_metric_ops, loss, fc1
            
        elif self.type == 'L':
            features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
            labels = tf.placeholder(tf.int64, [None,], name='labels')

            embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
            x = tf.nn.embedding_lookup(embs, features)
            
            
            if "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0" in self.masks: 
                mask = self.masks['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0'][2][1]
                n_hidden = int(np.count_nonzero(mask)/4)
            else:
                n_hidden = self.n_hidden
            stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(2)])
            outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)

            mask = self.masks['dense1/kernel:0'][2]
            new_size = np.count_nonzero(mask)
            fc1 = tf.layers.dense(inputs=outputs[:, -1, :], units=new_size, name='dense1')
            pred = tf.squeeze(tf.layers.dense(inputs=fc1, units=1))
            
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pred)
            grads_and_vars = self.optimizer.compute_gradients(loss)
            grads, _ = zip(*grads_and_vars)
            train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
            correct_pred = tf.equal(tf.to_int64(tf.greater(pred,0)), labels)
            eval_metric_ops = tf.count_nonzero(correct_pred)

            
            return features, labels, train_op, eval_metric_ops, loss, fc1


    def process_x(self, raw_x_batch, max_words=25):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [1 if e=='4' else 0 for e in raw_y_batch]
        # y_batch = [val_to_vec(self.num_classes, e) for e in y_batch]
        y_batch = np.array(y_batch)

        return y_batch

"""Bag-of-words logistic regression."""

import numpy as np
import os
import sys
import tensorflow as tf

from model import Model
from utils.model_utils import batch_data


class ClientModel(Model):

    def __init__(self, seed, lr, num_classes, input_dim, types='H', masks=None, optimizer=None):
        self.num_classes = 2#num_classes
        self.input_dim = 2#input_dim
        self.type = types
        self.masks = masks
        super(ClientModel, self).__init__(seed, lr, optimizer)

    def create_model(self):
        if self.type == 'H':
            features = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='features')
            labels = tf.placeholder(tf.int64, shape=[None,], name='labels')

            dense = tf.layers.dense(inputs=features, units=6, name='dense1')
            act_1 = tf.nn.relu(dense)
            logits = tf.layers.dense(inputs=dense, units=self.num_classes)

            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            predictions = tf.argmax(logits, axis=-1)
            correct_pred = tf.equal(predictions, labels)
            eval_metric_ops = tf.count_nonzero(correct_pred)
            
            grads_and_vars = self.optimizer.compute_gradients(loss)
            
            return features, labels, train_op, eval_metric_ops, tf.reduce_mean(loss), act_1
        elif self.type == 'L':
            features = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='features')
            labels = tf.placeholder(tf.int64, shape=[None,], name='labels')

            mask = self.masks['dense1/kernel:0'][2]
            new_size = np.count_nonzero(mask)
            dense = tf.layers.dense(inputs=features, units=new_size, name='dense1')
            act_1 = tf.nn.relu(dense)
            logits = tf.layers.dense(inputs=dense, units=self.num_classes)

            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            predictions = tf.argmax(logits, axis=-1)
            correct_pred = tf.equal(predictions, labels)
            eval_metric_ops = tf.count_nonzero(correct_pred)

            grads_and_vars = self.optimizer.compute_gradients(loss)
            
            return features, labels, train_op, eval_metric_ops, tf.reduce_mean(loss), act_1


    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def _run_epoch(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size, self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with self.graph.as_default():
                self.sess.run(
                    self.train_op,
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data})

    def _test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])

        with self.graph.as_default():
            tot_acc, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={
                    self.features: x_vecs, 
                    self.labels: labels
                })

        acc = float(tot_acc) / len(x_vecs)
        return {'accuracy': acc}

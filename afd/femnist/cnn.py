import tensorflow as tf

from model import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, types='H', masks=None, optimizer=None):
        self.num_classes = num_classes
        self.type = types
        self.masks = masks
        super(ClientModel, self).__init__(seed, lr, optimizer)


    def create_model(self):
        """Model function for CNN."""
        if self.type == 'H':
            features = tf.placeholder(
                tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
            labels = tf.placeholder(tf.int64, shape=[None], name='labels')
            input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name = "conv1" )
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name = "conv_last" )
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            dense = tf.layers.dense(inputs=pool2_flat, units=2048, name='dense1')
            act_1 = tf.nn.relu(dense)
            logits = tf.layers.dense(inputs=act_1, units=self.num_classes)
            predictions = {
              "classes": tf.argmax(input=logits, axis=1),
              "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            # TODO: Confirm that opt initialized once is ok?
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
            grads_and_vars = self.optimizer.compute_gradients(loss)
            return features, labels, train_op, eval_metric_ops, loss, act_1
        elif self.type == 'L':
            features = tf.placeholder(
                tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
            labels = tf.placeholder(tf.int64, shape=[None], name='labels')
            input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            mask = self.masks['conv1/kernel:0'][2]
            new_filters = np.count_nonzero(mask)
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=new_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name = "conv1" )
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            
            mask = self.masks['conv_last/kernel:0'][2]
            new_filters = np.count_nonzero(mask)
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=new_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name = "conv_last"  )
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * new_filters])
            
            mask = self.masks['dense1/kernel:0'][2]
            new_size = np.count_nonzero(mask)
            dense = tf.layers.dense(inputs=pool2_flat, units=new_size, name='dense1')
            act_1 = tf.nn.relu(dense)
            logits = tf.layers.dense(inputs=act_1, units=self.num_classes)
            predictions = {
              "classes": tf.argmax(input=logits, axis=1),
              "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            # TODO: Confirm that opt initialized once is ok?
            train_op = self.optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
            grads_and_vars = self.optimizer.compute_gradients(loss)
            return features, labels, train_op, eval_metric_ops, loss, act_1



    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from compressible_model import DTYPE, Compressible


class Lenet5(Compressible):
    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, padding='SAME', strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2, padding='SAME'):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding=padding)

    # Create model
    def conv_net(self, x, weights):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], weights['bc1'], padding='VALID')
        print(conv1.shape)
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2, padding='SAME')
        print(conv1.shape)
        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], weights['bc2'], padding='VALID')
        print(conv2.shape)
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2, padding='SAME')
        print(conv2.shape)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), weights['bd1'])
        fc1 = tf.nn.relu(fc1)

        # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), weights['bd2'])
        # fc2 = tf.nn.relu(fc2)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), weights['bout'])
        return out

    def __init__(self, bpb, load_name=None):
        super(Lenet5, self).__init__('Lenet5')
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # Training Parameters
        self.batch_size = 256

        # Network Parameters
        num_input = 784  # MNIST data input (img shape: 28*28)
        num_classes = 10  # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, num_input]) - 0.5
        self.Y = tf.placeholder(tf.float32, [None, num_classes])

        # Weights
        weight_names = ['wc1', 'wc2', 'wd1', 'out', 'bc1', 'bc2', 'bd1', 'bout']
        weight_dims = [[5, 5, 1, 20], [5, 5, 20, 50], [4 * 4 * 50, 500],
                       [500, num_classes], [20], [50], [500], [num_classes]]
        weight_hash_groups = [1, 2, 50, 1, 1, 1, 1, 1]
        weight_initializers = []
        for d in weight_dims:
            if len(d) == 4:
                weight_initializers.append(('normal', np.sqrt(1. / (d[0] * d[1] * d[2]))))
            else:
                weight_initializers.append(('normal', np.sqrt(1. / d[0])))

        weights = {}
        weights.update(zip(weight_names, self.initialize_variables(weight_dims,
                                                                   weight_initializers,
                                                                   weight_hash_groups,
                                                                   30, bpb,
                                                                   kl_penalty_step=1.0001)))
        # Construct model
        logits = self.conv_net(self.X, weights)

        # Evaluate model
        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, DTYPE))

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y)) + self.kl_loss

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.001,  # Base learning rate.
            global_step,  # Current index into the dataset.
            30 * self.mnist.train.images.shape[0] / self.batch_size,  # Decay step.
            1.,  # Decay rate.
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        no_scales_list = [v for v in tf.trainable_variables() if v is not self.p_scale_vars]
        assert len(no_scales_list) < len(tf.trainable_variables())
        self.train_op_no_scales = optimizer.minimize(self.loss, var_list=no_scales_list)

        self.initialize_session(load_name)

    def get_feed_dict(self, validation=False):
        if validation:
            batch_x, batch_y = self.mnist.validation.images, self.mnist.validation.labels
        else:
            batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
        return {self.X: batch_x, self.Y: batch_y}

    def get_train_op(self, training=True):
        if training:
            return self.train_op
        else:
            return self.train_op_no_scales

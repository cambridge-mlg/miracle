import os
import tensorflow as tf
import numpy as np
from compressible_model import DTYPE, Compressible
import cifar10_data
from cifar10_data import img_size, num_channels, num_classes

img_size_cropped = 32

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size + 8,
                                                       target_width=img_size + 8)
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images


class VGG(Compressible):
    def get_feed_dict(self, validation=False):
        if validation:
            idx = range(self.test_ind, self.test_ind + 1000)
            self.test_ind += 1000
            if self.test_ind == self.images_test.shape[0]:
                self.test_ind = 0

            # Use the random index to select random images and labels.
            batch_x = self.images_test[idx, :, :, :]
            batch_y = self.labels_test[idx, :]
        else:
            idx = np.random.choice(self.images_train.shape[0],
                                   size=self.batch_size,
                                   replace=False)

            # Use the random index to select random images and labels.
            batch_x = self.images_train[idx, :, :, :]
            batch_y = self.labels_train[idx, :]
        return {self.X: batch_x, self.Y: batch_y}

    def get_train_op(self, training):
        if training == 'training':
            return self.train_op
        elif training == 'compression':
            return self.train_op_no_scales
        elif training == 'pretrain':
            return self.pretrain_op

    def conv2d(self, x, W, b, padding='SAME', strides=1, training=True):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        x = tf.layers.batch_normalization(x, training=training)
        return tf.nn.relu(x)

    def fc(self, x, W, b):
        x = tf.add(tf.matmul(x, W), b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2, padding='SAME'):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding=padding)

    def main_network(self, images, training, conv_layers, weights):
        # print(conv_layers)
        # print(weights)
        for l in conv_layers:
            print(images.get_shape().as_list())
            if l == 'M':
                images = self.maxpool2d(images)
            else:
                images = self.conv2d(images, weights['w' + l], weights['b' + l], training=training)

        # print(images.shape)
        images = tf.reshape(images, [-1, weights['wfc1'].get_shape().as_list()[0]])
        # print(images.shape)
        images = tf.layers.dropout(images, training=training, rate=0.5 * (1.0 - self.enable_kl_loss))
        images = self.fc(images, weights['wfc1'], weights['bfc1'])
        images = tf.layers.dropout(images, training=training, rate=0.5 * (1.0 - self.enable_kl_loss))
        images = self.fc(images, weights['wfc2'], weights['bfc2'])
        logits = tf.add(tf.matmul(images, weights['wout']), weights['bout'])
        prediction = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        return prediction, loss

    def create_network(self, conv_layers, weights, training):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.

        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            images = self.X

            # Create TensorFlow graph for pre-processing.
            images = pre_process(images=images, training=training)

            # Create TensorFlow graph for the main processing.
            y_pred, loss = self.main_network(images, training, conv_layers, weights)

        return y_pred, loss

    def __init__(self, type='D', load_name=None, bits=5):
        super(VGG, self).__init__('VGG')
        cifar10_data.data_path = "/scratch/mh740/compression_models/data/cifar10/"
        cifar10_data.maybe_download_and_extract()

        images_train, cls_train, self.labels_train = cifar10_data.load_training_data()
        images_test, cls_test, self.labels_test = cifar10_data.load_test_data()
        self.test_ind = 0

        print('CIFAR10 Training set: {}, Test set: {}'.format(images_train.shape, images_test.shape))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.images_train = (images_train - mean) / std
        self.batch_size = 128
        self.images_test = (images_test - mean) / std

        self.X = tf.placeholder(DTYPE, shape=[None, img_size, img_size, num_channels])
        self.Y = tf.placeholder(DTYPE, shape=[None, num_classes])
        self.distorted_images = pre_process(images=self.X, training=True)


        layers = cfg[type]
        weight_names = []
        weight_dims = []
        initializers = []
        hash_group_sizes = []
        conv_hash_size = 1
        fc_hash_size = 1
        conv_layers = []
        i = 1
        in_channels = 3
        for l in layers:
            if l == 'M':
                conv_layers.append(l)
            else:
                conv_layers.append('conv' + str(i))
                weight_names.append('wconv' + str(i))
                weight_dims.append([3, 3, in_channels, l])
                initializers.append(('normal', np.sqrt(2. / (3*3*l))))
                if in_channels == 512:
                    hash_group_sizes.append(8)
                else:
                    hash_group_sizes.append(conv_hash_size)
                weight_names.append('bconv' + str(i))
                weight_dims.append([l])
                initializers.append(('zero', 0.))
                hash_group_sizes.append(1)
                in_channels = l
                i += 1

        weight_names.append('wfc1')
        weight_dims.append([512, 512])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(fc_hash_size)

        weight_names.append('bfc1')
        weight_dims.append([512])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(1)

        weight_names.append('wfc2')
        weight_dims.append([512, 512])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(fc_hash_size)

        weight_names.append('bfc2')
        weight_dims.append([512])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(1)

        weight_names.append('wout')
        weight_dims.append([512, 10])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(fc_hash_size)

        weight_names.append('bout')
        weight_dims.append([10])
        initializers.append(('uni', np.sqrt(1. / 512)))
        hash_group_sizes.append(1)

        weights = {}
        weights.update(zip(weight_names, self.initialize_variables(weight_dims, initializers, hash_group_sizes, 32, bits)))

        _, pred_loss = self.create_network(conv_layers, weights, training=True)

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            0.05,  # Base learning rate.
            global_step,  # Current index into the dataset.
            30 * images_train.shape[0] / self.batch_size,  # Decay step.
            0.5,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        momentum = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        trainable_variables_before = tf.trainable_variables()

        y_pred, pred_loss_notrain = self.create_network(conv_layers, weights, training=False)
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        y_true_cls = tf.argmax(self.Y, dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # print([v.name for v in tf.trainable_variables() if v not in trainable_variables_before])
        self.loss = pred_loss + self.kl_loss
        adam = tf.train.AdamOptimizer(learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.pretrain_op = momentum.minimize(pred_loss + self.weight_decay_loss, global_step=global_step)
            self.train_op = adam.minimize(self.loss)
            no_scales_list = [v for v in tf.trainable_variables() if v is not self.p_scale_vars]
            assert len(no_scales_list) < len(tf.trainable_variables())
            self.train_op_no_scales = adam.minimize(self.loss, var_list=no_scales_list)

        self.initialize_session(load_name)
        for w in weights:
            # print(w, np.max(self.sess.run(weights[w])))

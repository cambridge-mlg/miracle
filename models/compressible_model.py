import tensorflow as tf
import os
import numpy as np
from scipy.stats import norm
import sobol_seq

DTYPE = tf.float32


class Compressible(object):
    def __init__(self, name, message_freq=1000):
        self.name = name
        self.message_freq = message_freq
        self.message_counter = 0

    def get_feed_dict(self, validation=False):
        raise NotImplementedError

    def get_train_op(self, training):
        raise NotImplementedError

    def training_step(self, training, extra_ops):
        self.message_counter += 1
        if self.message_counter % self.message_freq == 0 or self.message_counter == 1:
            loss, training_acc, kl, kl_loss = self.sess.run([self.loss, self.accuracy, self.mean_kl, self.kl_loss],
                                                   feed_dict=self.get_feed_dict())

            mean_validation_acc = 0.0
            for i in range(10):
                validation_acc = self.sess.run(self.accuracy,
                                               feed_dict=self.get_feed_dict(validation=True))
                mean_validation_acc += validation_acc
            mean_validation_acc /= 10.
            print("Iteration {}, Validation = {}, Training = {}, Loss = {}, KL-Loss = {}, KL_2 = {}".format(
                self.message_counter,
                mean_validation_acc,
                training_acc,
                loss,
                kl_loss,
                kl / np.log(2.)))
            path = '/scratch/mh740/compression_models/{}/{}/{}.ckpt'.format(self.name, training,
                                                                         self.message_counter)
            if not os.path.exists(path):
                os.makedirs(path)
            self.saver.save(self.sess, path)

        self.sess.run((self.get_train_op(training=training), extra_ops),
                      feed_dict=self.get_feed_dict())

    def train(self, iterations, enforce_kl):
        if enforce_kl:
            self.sess.run(self.enable_kl_loss.assign(1.))
            with tf.control_dependencies([self.get_train_op(training='training')]):
                extra_ops = [tf.identity(self.kl_penalty_update)]
            for i in range(iterations):
                self.training_step(training='training', extra_ops=extra_ops)
        else:
            self.sess.run(self.enable_kl_loss.assign(0.))
            #with tf.control_dependencies([self.get_train_op(training='pretrain')]):
            #    extra_ops = [tf.identity(self.weight_decay_op)]
            for i in range(iterations):
                self.training_step(training='pretrain', extra_ops=[])#extra_ops)


        mean_validation_acc = 0.0
        for i in range(20):
            validation_acc = self.sess.run(self.accuracy,
                                           feed_dict=self.get_feed_dict(validation=True))
        mean_validation_acc += validation_acc
        return mean_validation_acc / 20.

    def compress(self, retrain_iter, kl_penalty_step=1.0005):
        self.sess.run(self.kl_penalty_step.assign(kl_penalty_step))
        n_blocks = self.fixed_weights.get_shape().as_list()[0]
        self.sess.run(self.enable_kl_loss.assign(1.))
        for i in range(n_blocks):
            self.sess.run(self.comp_ops, feed_dict={self.block_to_comp: i})
            print('Block {} of {} compressed'.format(i, n_blocks))
            for j in range(retrain_iter):
                self.training_step(training='compression', extra_ops=self.kl_penalty_update)

        mean_validation_acc = 0.0
        for i in range(100):
            validation_acc = self.sess.run(self.accuracy,
                                           feed_dict=self.get_feed_dict(validation=True))
            mean_validation_acc += validation_acc
        return mean_validation_acc / 100.

    def initialize_variables(self,
                             dimensions,
                             initializers,
                             hash_group_sizes,
                             block_size,
                             bits_per_block,
                             weight_decay=5e-4,
                             kl_penalty_step=1.00005):
        assert len(initializers) == len(dimensions)
        num_vars = 0
        for dim, group_size in zip(dimensions, hash_group_sizes):
            assert np.prod(dim) % group_size == 0
            num_vars += np.prod(dim) / group_size
        n_blocks = 1 + (num_vars - 1) / block_size
        shape = [n_blocks, block_size]
        print('Number of blocks: {}, Block size: {}, Bits per block: {}, Target KL: {}, Overall bits {}, Ratio: {}'.format(
            n_blocks, block_size, bits_per_block, bits_per_block, bits_per_block*n_blocks, np.sum([np.prod(dim) for dim in dimensions])*32. / (bits_per_block * n_blocks)
        ))
        num_vars_ub = np.prod(shape)

        np.random.seed(420)
        permutation = np.random.permutation(num_vars_ub)
        permutation_inv = np.argsort(permutation)
        var_sizes = [np.prod(dim)/group_size for dim, group_size in zip(dimensions, hash_group_sizes)]
        # print(var_sizes)
        # print(initializers)

        self.p_scale_vars = tf.Variable(tf.fill([len(dimensions) + 1], -2.), dtype=DTYPE)
        p_perm_inv = np.repeat(range(len(dimensions) + 1), var_sizes + [num_vars_ub - num_vars])[permutation_inv]
        self.p_scale = tf.reshape(tf.gather(tf.exp(self.p_scale_vars), p_perm_inv), (shape))
        p = tf.contrib.distributions.Normal(loc=0., scale=self.p_scale)
        mu_init_list = []
        for (type, val), size in zip(initializers, var_sizes):
            if type == 'normal':
                mu_init_list.append(np.random.normal(size=size, loc=0., scale=val))
            elif type == 'uni':
                mu_init_list.append(np.random.uniform(-val, val, size=size))
            elif type == 'zero':
                mu_init_list.append(np.zeros(size))
            else:
                assert False

        mu_init = np.concatenate(mu_init_list)
        # print(num_vars, mu_init.shape)
        # print(var_sizes, [init.shape for init in mu_init_list])
        init_inv_permuted = np.concatenate((mu_init,
                                            np.zeros(num_vars_ub - num_vars)),
                                           axis=0)[permutation_inv]

        mu = tf.Variable(init_inv_permuted.reshape(shape), dtype=DTYPE, name='mu')
        self.mu = mu
        self.weight_decay_loss = tf.reduce_sum(tf.square(mu)) * weight_decay
        self.sigma_var = tf.Variable(tf.fill(shape, tf.cast(-10., dtype=DTYPE, name='sigma')))
        sigma = tf.exp(self.sigma_var)
        self.sigma = sigma
        epsilon = tf.random_normal(shape)
        self.w_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
        variational_weights = mu + epsilon * sigma
        self.fixed_weights = tf.Variable(tf.zeros_like(variational_weights), trainable=False)
        self.mask = tf.Variable(tf.ones([n_blocks]), trainable=False)
        kl_penalties = tf.Variable(tf.fill([n_blocks], tf.cast(1e-8, dtype=DTYPE)), trainable=False)
        self.kl_penalties = kl_penalties

        kl_target = tf.Variable(bits_per_block * np.log(2.), dtype=tf.float32, trainable=False)
        block_kl = tf.reduce_sum(tf.distributions.kl_divergence(self.w_dist, p), axis=1)
        self.mean_kl = tf.reduce_mean(block_kl)

        self.enable_kl_loss = tf.Variable(1., dtype=DTYPE, trainable=False)
        self.kl_loss = tf.reduce_sum(block_kl * self.mask * kl_penalties) * self.enable_kl_loss
        self.kl_penalty_step = tf.Variable(kl_penalty_step, trainable=False)
        self.kl_penalty_update = [kl_penalties.assign(tf.where(tf.logical_and(tf.cast(self.mask, tf.bool),
                                                                              tf.greater(block_kl, kl_target)),
                                                               kl_penalties * self.kl_penalty_step,
                                                               kl_penalties / self.kl_penalty_step))]

        mask_expanded = tf.expand_dims(self.mask, 1)
        combined_weights = tf.reshape(mask_expanded * variational_weights
                                      + (1. - mask_expanded) * self.fixed_weights,
                                      [-1])

        permuted_weights = tf.gather(combined_weights, permutation)
        split_weights = tf.split(permuted_weights, var_sizes + [num_vars_ub - num_vars])

        result = []
        i = 0
        for dim in dimensions:
            split = tf.expand_dims(split_weights[i], axis=1) * np.random.choice([-1., 1.], size=hash_group_sizes[i])
            # print(split.get_shape().as_list())
            result.append(tf.reshape(split, dim))
            i += 1

        self.initialize_compressor(bits_per_block)
        return result

    def initialize_compressor(self, bits_per_block):
        with tf.variable_scope('compressor'):
            self.block_to_comp = tf.placeholder(tf.int32)
            shape = self.fixed_weights.get_shape().as_list()
            # block_ind = tf.expand_dims(block, 1)
            # mask = tf.scatter_nd(block_ind, tf.ones([1], dtype=tf.float32), shape)
            # sample_shape = tf.concat(([tries], shape), axis=0)

            # sequencer = ghalton.Halton(shape[1])
            n_blocks = shape[0]
            sobol_dim = shape[1]
            assert sobol_dim <= 40
            uni_quasi = np.array(sobol_seq.i4_sobol_generate(sobol_dim, np.power(2, bits_per_block), skip=1)).transpose()
            normal_quasi = norm.ppf(uni_quasi).transpose()
            #normal_quasi = np.tile(normal_quasi[:, None, :], [1, n_blocks, 1])
            # This line helps but not exactly sure why
            # normal_quasi /= np.sqrt(np.mean(np.square(normal_quasi), axis=1))[:, None]
            sample_block = tf.constant(normal_quasi, dtype=DTYPE)

            # normal = np.random.normal(size=(tries, shape[0], shape[1]))
            # normal /= np.sqrt(np.mean(np.square(normal), axis=2))[:, :, None]
            # sample_block = tf.constant(normal, dtype=tf.float32) * p_scale

            block_p = self.p_scale[self.block_to_comp, :]
            block_mu = self.mu[self.block_to_comp, :]
            block_sigma = self.sigma[self.block_to_comp, :]
            nll_q = tf.reduce_sum(tf.square(block_mu - sample_block * block_p) / (2*tf.square(block_sigma)), axis=1)
            nll_p = tf.reduce_sum(tf.square(sample_block), axis=1)
            #prob = tf.Print(tf.exp(-nll), [nll_q, nll_p], summarize=100)
            #norm_prob = tf.Print(prob / tf.reduce_sum(prob), [prob], summarize=100)
            dist = tf.distributions.Categorical(probs=tf.nn.softmax(nll_p - nll_q)) # , validate_args=True) #Risky
            index = dist.sample([])

            # This line makes the algorithm objectively better. But we cannot prove it theoretically.
            # min_index = tf.argmin(nll_q, axis=0)

            best_sample = sample_block[index, :] * block_p
            self.comp_ops = []
            self.comp_ops.append(tf.scatter_update(self.fixed_weights,
                                                   [self.block_to_comp],
                                                   [best_sample]))
            self.comp_ops.append(tf.scatter_update(self.mask, [self.block_to_comp], [0.]))

    def initialize_session(self, load_name=None):
        # Initialize the variables (i.e. assign their default value)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.loader = tf.train.Saver(var_list=[v for v in tf.all_variables() if v not in []])
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        # Run the initializer
        self.sess.run(init)

        if load_name is not None:
            # tf.reset_default_graph()
            path = '/scratch/mh740/compression_models/{}'.format(load_name)
            self.loader.restore(self.sess, path)

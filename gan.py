"""Main GAN module."""

import tensorflow as tf
import numpy as np
import time
import os
import pickle
import utils
import metrics
import itertools
import math
from colorize import colorize

from plot_summary import PlotSummaryLog
from default import default_params, default_params_cosmology, default_params_time


class GAN(object):
    """General (Generative Adversarial Network) GAN class.

    This class contains all the method to train an use a GAN. Note that the
    model, i.e. the architecture of the network is not handled by this class.
    """

    def __init__(self, params, model=None, is_3d=False):
        """Build the GAN network.

        Input arguments
        ---------------
        * params : structure of parameters
        * model  : model class for the architecture of the network
        * is_3d  : (To be removed soon)

        Please refer to the module `model` for details about
        the requirements of the class model.
        """
        tf.reset_default_graph()

        self.params = default_params(params)
        self._is_3d = is_3d
        if model is None:
            model = params['model']
        else:
            params['model'] = model
        self._savedir = params['save_dir']
        self._sess = None
        self.batch_size = self.params['optimization']['batch_size']
        self._prior_distribution = self.params['prior_distribution']
        self._mean = tf.get_variable(
            name="mean",
            dtype=tf.float32,
            shape=[1],
            trainable=False,
            initializer=tf.constant_initializer(0.))
        self._var = tf.get_variable(
            name="var",
            dtype=tf.float32,
            shape=[1],
            trainable=False,
            initializer=tf.constant_initializer(1.))

        self._z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        if is_3d:
            if len(self.params['image_size']) == 3:
                shape = [None, *self.params['image_size'], 1]
            else:
                shape = [None, *self.params['image_size']]
        else:
            if len(self.params['image_size']) == 2:
                # TODO Clean this
                if 'time' in self.params.keys():
                    shape = [None, *self.params['image_size'], params['time']['num_classes']]
                else:
                    shape = [None, *self.params['image_size'], 1]
            else:
                shape = [None, *self.params['image_size']]

        self._X = tf.placeholder(tf.float32, shape=shape, name='X')

        name = params['name']
        self._model = model(
            params,
            self._normalize(self._X),
            self._z,
            name=name if name else None,
            is_3d=is_3d)
        self._model_name = self._model.name
        self._D_loss = self._model.D_loss
        self._G_loss = self._model.G_loss
        self._G_fake = self._unnormalize(self._model.G_fake)

        t_vars = tf.trainable_variables()
        utils.show_all_variables()

        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        e_vars = [var for var in t_vars if 'encoder' in var.name]

        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer_D, optimizer_G, optimizer_E = self._build_optmizer()

        grads_and_vars_d = optimizer_D.compute_gradients(
            self._D_loss, var_list=d_vars)
        grads_and_vars_g = optimizer_G.compute_gradients(
            self._G_loss, var_list=g_vars)

        self._D_solver = optimizer_D.apply_gradients(
            grads_and_vars_d, global_step=global_step)
        self._G_solver = optimizer_G.apply_gradients(
            grads_and_vars_g, global_step=global_step)

        if self.has_encoder:
            self._E_loss = self._model.E_loss
            grads_and_vars_e = optimizer_E.compute_gradients(
                self._E_loss, var_list=e_vars)
            self._E_solver = optimizer_E.apply_gradients(
                grads_and_vars_e, global_step=global_step)

        self._buid_opt_summaries(optimizer_D, grads_and_vars_d, optimizer_G,
                                 grads_and_vars_g, optimizer_E)

        # Summaries
        self._build_image_summary()

        tf.summary.histogram('Prior/z', self._z, collections=['Images'])

        self.summary_op = tf.summary.merge(tf.get_collection("Training"))
        self.summary_op_img = tf.summary.merge(tf.get_collection("Images"))

        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    def _build_image_summary(self):
        if self.is_3d:
            tile_shape = utils.get_tile_shape_from_3d_image(
                self.params['image_size'])

            self.real_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[5, *tile_shape, 1], name='real_placeholder')

            self.fake_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[5, *tile_shape, 1], name='fake_placeholder')

            self.summary_op_real_image = tf.summary.image(
                "training/plot_real", 
                self.real_placeholder,
                max_outputs=5)

            self.summary_op_fake_image = tf.summary.image(
                "training/plot_fake", 
                self.fake_placeholder,
                max_outputs=5)

            if self.normalized():  # displaying only one slice from the normalized 3d image
                tf.summary.image(
                    "training/Real_Image_normalized", (self._normalize(
                        self._X))[:, 1, :, :, :],
                    max_outputs=4,
                    collections=['Images'])

                tf.summary.image(
                    "training/Fake_Image_normalized", (self._normalize(
                        self._G_fake))[:, 1, :, :, :],
                    max_outputs=4,
                    collections=['Images'])
        else:
            vmin = tf.reduce_min(self._X)
            vmax = tf.reduce_max(self._X)
            tf.summary.image(
                "training/Real_Image",
                colorize(self._X, vmin, vmax),
                max_outputs=4,
                collections=['Images'])
            tf.summary.image(
                "training/Fake_Image",
                colorize(self._G_fake, vmin, vmax),
                max_outputs=4,
                collections=['Images'])
            if self.normalized():
                tf.summary.image(
                    "training/Real_Image_normalized",
                    self._normalize(self._X),
                    max_outputs=4,
                    collections=['Images'])
                tf.summary.image(
                    "training/Fake_Image_normalized",
                    self._normalize(self._G_fake),
                    max_outputs=4,
                    collections=['Images'])

    def _build_optmizer(self):

        gen_learning_rate = self.params['optimization']['gen_learning_rate']
        enc_learning_rate = self.params['optimization']['enc_learning_rate']
        disc_learning_rate = self.params['optimization']['disc_learning_rate']
        gen_optimizer = self.params['optimization']['gen_optimizer']
        disc_optimizer = self.params['optimization']['disc_optimizer']
        beta1 = self.params['optimization']['beta1']
        beta2 = self.params['optimization']['beta2']
        epsilon = self.params['optimization']['epsilon']

        if gen_optimizer == "adam":
            optimizer_G = tf.train.AdamOptimizer(
                learning_rate=gen_learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon)
            if self.has_encoder:
                optimizer_E = tf.train.AdamOptimizer(
                    learning_rate=enc_learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon)
            else:
                optimizer_E = None
        elif gen_optimizer == "rmsprop":
            optimizer_G = tf.train.RMSPropOptimizer(
                learning_rate=gen_learning_rate)
            if self.has_encoder:
                optimizer_E = tf.train.RMSPropOptimizer(
                    learning_rate=enc_learning_rate)
            else:
                optimizer_E = None
        elif gen_optimizer == "sgd":
            optimizer_G = tf.train.GradientDescentOptimizer(
                learning_rate=gen_learning_rate)
            if self.has_encoder:
                optimizer_E = tf.train.GradientDescentOptimizer(
                    learning_rate=enc_learning_rate)
            else:
                optimizer_E = None
        else:
            raise Exception(" [!] Choose optimizer between [adam,rmsprop,sgd]")

        if disc_optimizer == "adam":
            optimizer_D = tf.train.AdamOptimizer(
                learning_rate=disc_learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon)
        elif disc_optimizer == "rmsprop":
            optimizer_D = tf.train.RMSPropOptimizer(
                learning_rate=disc_learning_rate)
        elif disc_optimizer == "sgd":
            optimizer_D = tf.train.GradientDescentOptimizer(
                learning_rate=disc_learning_rate)
        else:
            raise Exception(" [!] Choose optimizer between [adam,rmsprop]")

        return optimizer_D, optimizer_G, optimizer_E

    def _buid_opt_summaries(self, optimizer_D, grads_and_vars_d, optimizer_G,
                            grads_and_vars_g, optimizer_E):

        grad_norms_d = [
            tf.sqrt(tf.nn.l2_loss(g[0]) * 2) for g in grads_and_vars_d
        ]
        grad_norm_d = [tf.reduce_sum(grads) for grads in grad_norms_d]
        final_grad_d = tf.reduce_sum(grad_norm_d)
        tf.summary.scalar(
            "Disc/Gradient_Norm", final_grad_d, collections=["Training"])

        grad_norms_g = [
            tf.sqrt(tf.nn.l2_loss(g[0]) * 2) for g in grads_and_vars_g
        ]
        grad_norm_g = [tf.reduce_sum(grads) for grads in grad_norms_g]
        final_grad_g = tf.reduce_sum(grad_norm_g)
        tf.summary.scalar(
            "Gen/Gradient_Norm", final_grad_g, collections=["Training"])

        gen_learning_rate = self.params['optimization']['gen_learning_rate']
        enc_learning_rate = self.params['optimization']['enc_learning_rate']
        disc_learning_rate = self.params['optimization']['disc_learning_rate']
        gen_optimizer = self.params['optimization']['gen_optimizer']
        disc_optimizer = self.params['optimization']['disc_optimizer']

        def get_lr_ADAM(optimizer, learning_rate):
            beta1_power, beta2_power = optimizer._get_beta_accumulators()
            optim_learning_rate = (learning_rate * tf.sqrt(1 - beta2_power) /
                                   (1 - beta1_power))

            return optim_learning_rate

        if gen_optimizer == "adam":
            optim_learning_rate_G = get_lr_ADAM(optimizer_G, gen_learning_rate)
            tf.summary.scalar(
                'Gen/ADAM_learning_rate',
                optim_learning_rate_G,
                collections=["Training"])

            if optimizer_E is not None:
                optim_learning_rate_E = get_lr_ADAM(optimizer_E, enc_learning_rate)
                tf.summary.scalar(
                    'Gen/ADAM_learning_rate',
                    optim_learning_rate_E,
                    collections=["Training"])

        if disc_optimizer == "adam":
            optim_learning_rate_D = get_lr_ADAM(optimizer_D, disc_learning_rate)
            tf.summary.scalar(
                'Disc/ADAM_learning_rate',
                optim_learning_rate_D,
                collections=["Training"])

    def add_input_channel(self, X):
        '''
        X: input tensor containing real data
        '''
        if self._is_3d:
            if len(X.shape) == 4: # (batch_size, x, y, z)
                X = X.reshape([*X.shape, 1])
        else:
            if (len(X.shape) == 3): # (batch_size, x, y)
                X = X.reshape([*X.shape, 1])

        return X

    def train(self, dataset, resume=False):

        n_data = dataset.N

        self._counter = 1
        self._n_epoch = self.params['optimization']['epoch']
        self._total_iter = self._n_epoch * (n_data // self.batch_size) - 1
        self._n_batch = n_data // self.batch_size

        self._save_current_step = False

        # Create the save diretory if it does not exist
        os.makedirs(self.params['save_dir'], exist_ok=True)
        run_config = tf.ConfigProto()

        with tf.Session(config=run_config) as self._sess:
            if resume:
                print('Load weights in the nework')
                self.load()
            else:
                self._sess.run(tf.global_variables_initializer())
                utils.saferm(self.params['summary_dir'])
                utils.saferm(self.params['save_dir'])
            if self.normalized():
                X = dataset.get_all_data()
                m = np.mean(X)
                v = np.var(X - m)
                self._mean.assign([m]).eval()
                self._var.assign([v]).eval()
            self._var.eval()
            self._mean.eval()
            self._summary_writer = tf.summary.FileWriter(
                self.params['summary_dir'], self._sess.graph)
            try:
                epoch = 0
                start_time = time.time()
                prev_iter_time = start_time

                while epoch < self._n_epoch:
                    for idx, batch_real in enumerate(
                            dataset.iter(self.batch_size)):

                        # print("batch_real shape:")
                        # print(tf.shape(batch_real)[0])
                        # print(tf.shape(batch_real)[1])
                        # print(tf.shape(batch_real)[2])
                        # print(tf.shape(batch_real)[3])
                        # print("test")

                        if resume:
                            # epoch = self.params['curr_epochs']
                            # idx = self.params['curr_idx']
                            self._counter = self.params['curr_counter']
                            resume = False
                        else:
                            # self.params['curr_epochs'] = epoch
                            # self.params['curr_idx'] = idx
                            self.params['curr_counter'] = self._counter

                        # reshape input according to 2d, 3d, or patch case
                        X_real = self.add_input_channel(batch_real)
                        for _ in range(self.params['optimization']['n_critic']):
                            sample_z = self._sample_latent(self.batch_size)
                            _, loss_d = self._sess.run(
                                [self._D_solver, self._D_loss],
                                feed_dict=self._get_dict(sample_z, X_real))
                            if self.has_encoder:
                                _, loss_e = self._sess.run(
                                    [self._E_solver, self._E_loss],
                                    feed_dict=self._get_dict(
                                        sample_z, X_real))

                        sample_z = self._sample_latent(self.batch_size)
                        _, loss_g, v, m = self._sess.run(
                            [
                                self._G_solver, self._G_loss, self._var,
                                self._mean
                            ],
                            feed_dict=self._get_dict(sample_z, X_real))

                        if np.mod(self._counter,
                                  self.params['print_every']) == 0:
                            current_time = time.time()
                            print("Epoch: [{:2d}] [{:4d}/{:4d}] "
                                  "Counter:{:2d}\t"
                                  "({:4.1f} min\t"
                                  "{:4.3f} examples/sec\t"
                                  "{:4.2f} sec/batch)\t"
                                  "L_Disc:{:.8f}\t"
                                  "L_Gen:{:.8f}".format(
                                      epoch, idx, self._n_batch,
                                      self._counter,
                                      (current_time - start_time) / 60,
                                      100.0 * self.batch_size /
                                      (current_time - prev_iter_time),
                                      (current_time - prev_iter_time) / 100,
                                      loss_d, loss_g))
                            prev_iter_time = current_time

                        self._train_log(
                            self._get_dict(sample_z, X_real))

                        if (np.mod(self._counter, self.params['save_every'])
                                == 0) | self._save_current_step:
                            self._save(self._savedir, self._counter)
                            self._save_current_step = False
                        self._counter += 1
                    epoch += 1
            except KeyboardInterrupt:
                pass
            self._save(self._savedir, self._counter)

    def _train_log(self, feed_dict):
        if np.mod(self._counter, self.params['viz_every']) == 0:
            summary_str, real_arr, fake_arr = self._sess.run(
                [self.summary_op_img, self._X, self._G_fake],
                feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_str, self._counter)

            # -- display cube by tiling square slices --
            if self.is_3d:
                real_summary, fake_summary = self._sess.run(
                    [self.summary_op_real_image, self.summary_op_fake_image],
                    feed_dict={
                        self.real_placeholder:
                        utils.tile_cube_slices(real_arr[:5, :, :, :, 0]),
                        self.fake_placeholder:
                        utils.tile_cube_slices(fake_arr[:5, :, :, :, 0])
                    })

                self._summary_writer.add_summary(real_summary, self._counter)
                self._summary_writer.add_summary(fake_summary, self._counter)
            # -----------------------------------------

        if np.mod(self._counter, self.params['sum_every']) == 0:
            summary_str = self._sess.run(self.summary_op, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_str, self._counter)

    def _sample_latent(self, bs=None):
        if bs is None:
            bs = self.batch_size
        latent_dim = self.params['generator']['latent_dim']
        return utils.sample_latent(bs, latent_dim, self._prior_distribution)

    def _get_dict(self, z=None, X=None, index=None, **kwargs):
        feed_dict = dict()
        if z is not None:
            if index is not None:
                feed_dict[self._z] = z[index]
            else:
                feed_dict[self._z] = z
        if X is not None:
            if index is not None:
                feed_dict[self._X] = X[index]
            else:
                feed_dict[self._X] = X
        for key, value in kwargs.items():
            if value is not None:
                if index is not None:
                    feed_dict[getattr(self._model, key)] = value[index]
                else:
                    feed_dict[getattr(self._model, key)] = value

        return feed_dict

    def _save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, self._model_name),
            global_step=step)
        self._save_obj()
        print('Model saved!')

    def _save_obj(self):
        # Saving the objects:
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'], exist_ok=True)

        path_param = os.path.join(self.params['save_dir'], 'params.pkl')
        with open(path_param, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, sess=None, checkpoint=None):
        """
        Given checkpoint, load the model.
        By default, load the latest model saved.
        """
        if checkpoint:
            file_name = os.path.join(
                self._savedir,
                self._model_name + '-' + str(checkpoint))
        else:
            file_name = None

        if sess:
            self._sess = sess
        elif self._sess is None:
            raise ValueError("Session not available at the time of loading model!")

        print(" [*] Reading checkpoints...")
        if file_name:
            self._saver.restore(self._sess, file_name)
            return True

        checkpoint_dir = self._savedir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            return True

        return False

    def generate(self,
                 N=None,
                 z=None,
                 X=None,
                 sess=None,
                 checkpoint=None,
                 **kwargs):
        """Generate new samples.

        The user can chose between different options depending on the model.

        **kwargs contains all possible optional arguments defined in the model.

        Arguments
        ---------
        * N : number of sample (Default None)
        * z : latent variable (Default None)
        * X : training image (Default None)
        * sess : tensorflow Session (Default None)
        * checkpoint : number of the checkpoint (Default None)
        * kwargs : keywords arguments that are defined in the model
        """

        if N and z:
            ValueError('Please choose between N and z')
        if sess is not None:
            self._sess = sess
            return self._generate_sample(
                N=N, z=z, X=X, **kwargs)

        with tf.Session() as self._sess:
            self.load(checkpoint=checkpoint)

            return self._generate_sample(
                N=N, z=z, X=X, **kwargs)

    def _generate_sample(self,
                         N=None,
                         z=None,
                         X=None,
                         **kwargs):

        if z is None:
            if N is None:
                N = self.batch_size
            z = self._sample_latent(N)
        return self._generate_sample_safe(z=z, X=X, **kwargs)

    def _get_sample_args(self, **kwargs):
        return self._G_fake

    def _special_vstack(self, gi):
        if type(gi[0]) is np.ndarray:
            return np.vstack(gi)
        else:
            s = []
            for j in range(len(gi[0])):
                s.append(np.vstack([el[j] for el in gi]))
            return tuple(s)

    def _generate_sample_safe(self, z=None, X=None, **kwargs):
        gen_images = []
        N = len(z)
        sind = 0
        bs = self.batch_size
        if N > bs:
            nb = (N - 1) // bs
            for i in range(nb):
                feed_dict = self._get_dict(
                    z=z, X=X, index=slice(sind, sind + bs), **kwargs)
                gi = self._sess.run(
                    self._get_sample_args(**kwargs), feed_dict=feed_dict)
                gen_images.append(gi)
                sind = sind + bs
        feed_dict = self._get_dict(z=z, X=X, index=slice(sind, N), **kwargs)
        gi = self._sess.run(self._get_sample_args(**kwargs), feed_dict=feed_dict)
        gen_images.append(gi)

        return self._special_vstack(gen_images)

    def _normalize(self, x):
        return (x - self._mean) / self._var

    def _unnormalize(self, x):
        return x * self._var + self._mean

    def normalized(self):
        return self.params['normalize']

    @property
    def has_encoder(self):
        return self._model.has_encoder

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_3d(self):
        return self._is_3d

    @property
    def average_over_all_channels(self):
        return False


class CosmoGAN(GAN):
    def __init__(self, params, model=None, is_3d=False):
        self.params = default_params_cosmology(params)
        super().__init__(params=self.params, model=model, is_3d=is_3d)

        self._backward_map = params['cosmology']['backward_map']
        self._forward_map = params['cosmology']['forward_map']

        # TODO: Make a variable to contain the clip max
        # tf.variable
        # self._G_raw = utils.inv_pre_process(self._G_fake,
        #                                     self.params['cosmology']['k'],
        #                                     scale=self.params['cosmology']['map_scale'])
        # self._X_raw = utils.inv_pre_process(self._X,
        #                                     self.params['cosmology']['k'],
        #                                     scale=self.params['cosmology']['map_scale'])

        self._md, self._plots = CosmoGAN._init_logs('Metrics')

        tf.summary.histogram(
            "Pixel/Fake", self._G_fake, collections=['Metrics'])
        tf.summary.histogram("Pixel/Real", self._X, collections=['Metrics'])

        self.summary_op_metrics = tf.summary.merge(
            tf.get_collection("Metrics"))

    @staticmethod
    def _init_logs(collection, name_suffix=''):
        """Initializes summary logs under the collection parameter name
        Parameter name suffix is added to all summary names

        :return dict with summary tensors
        :return dict with summary plots
        """
        md = dict()

        md['descriptives'] = tf.placeholder(
            tf.float64, shape=[2, 5], name="DescriptiveStatistics")

        tf.summary.scalar(
            "descriptives/mean_Fake" + name_suffix,
            md['descriptives'][0, 0],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/var_Fake" + name_suffix,
            md['descriptives'][0, 1],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/min_Fake" + name_suffix,
            md['descriptives'][0, 2],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/max_Fake" + name_suffix,
            md['descriptives'][0, 3],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/median_Fake" + name_suffix,
            md['descriptives'][0, 4],
            collections=[collection])

        tf.summary.scalar(
            "descriptives/mean_Real" + name_suffix,
            md['descriptives'][1, 0],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/var_Real" + name_suffix,
            md['descriptives'][1, 1],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/min_Real" + name_suffix,
            md['descriptives'][1, 2],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/max_Real" + name_suffix,
            md['descriptives'][1, 3],
            collections=[collection])
        tf.summary.scalar(
            "descriptives/median_Real" + name_suffix,
            md['descriptives'][1, 4],
            collections=[collection])

        md['peak_fake'] = tf.placeholder(
            tf.float64, shape=[None], name="peak_fake" + name_suffix)
        md['peak_real'] = tf.placeholder(
            tf.float64, shape=[None], name="peak_real" + name_suffix)
        tf.summary.histogram(
            "Peaks/Fake_log" + name_suffix, md['peak_fake'], collections=[collection])
        tf.summary.histogram(
            "Peaks/Real_log" + name_suffix, md['peak_real'], collections=[collection])

        md['distance_peak_comp'] = tf.placeholder(
            tf.float64, name='distance_peak_comp' + name_suffix)
        md['distance_peak_fake'] = tf.placeholder(
            tf.float64, name='distance_peak_fake' + name_suffix)
        md['distance_peak_real'] = tf.placeholder(
            tf.float64, name='distance_peak_real' + name_suffix)

        tf.summary.scalar(
            "Peaks/Ch2_Fake-Real" + name_suffix,
            md['distance_peak_comp'],
            collections=[collection])
        tf.summary.scalar(
            "Peaks/Ch2_Fake-Fake" + name_suffix,
            md['distance_peak_fake'],
            collections=[collection])
        tf.summary.scalar(
            "Peaks/Ch2_Real-Real" + name_suffix,
            md['distance_peak_real'],
            collections=[collection])

        md['cross_ps'] = tf.placeholder(
            tf.float64, shape=[3], name='cross_ps' + name_suffix)

        tf.summary.scalar(
            "PSD/Cross_Fake-Real" + name_suffix,
            md['cross_ps'][0],
            collections=[collection])
        tf.summary.scalar(
            "PSD/Cross_Fake-Fake" + name_suffix,
            md['cross_ps'][1],
            collections=[collection])
        tf.summary.scalar(
            "PSD/Cross_Real-Real" + name_suffix,
            md['cross_ps'][2],
            collections=[collection])

        md['l2_psd'] = tf.placeholder(tf.float32, name='l2_psd' + name_suffix)
        md['log_l2_psd'] = tf.placeholder(tf.float32, name='log_l2_psd' + name_suffix)
        md['l1_psd'] = tf.placeholder(tf.float32, name='l1_psd' + name_suffix)
        md['log_l1_psd'] = tf.placeholder(tf.float32, name='log_l1_psd' + name_suffix)
        tf.summary.scalar(
            "PSD/l2" + name_suffix, md['l2_psd'], collections=[collection])
        tf.summary.scalar(
            "PSD/log_l2" + name_suffix, md['log_l2_psd'], collections=[collection])
        tf.summary.scalar(
            "PSD/l1" + name_suffix, md['l1_psd'], collections=[collection])
        tf.summary.scalar(
            "PSD/log_l1" + name_suffix, md['log_l1_psd'], collections=[collection])

        md['l2_peak_hist'] = tf.placeholder(
            tf.float32, name='l2_peak_hist' + name_suffix)
        md['log_l2_peak_hist'] = tf.placeholder(
            tf.float32, name='log_l2_peak_hist' + name_suffix)
        md['l1_peak_hist'] = tf.placeholder(
            tf.float32, name='l1_peak_hist' + name_suffix)
        md['log_l1_peak_hist'] = tf.placeholder(
            tf.float32, name='log_l1_peak_hist' + name_suffix)
        tf.summary.scalar(
            "PEAK_HIST/l2" + name_suffix, md['l2_peak_hist'], collections=[collection])
        tf.summary.scalar(
            "PEAK_HIST/log_l2" + name_suffix,
            md['log_l2_peak_hist'],
            collections=[collection])
        tf.summary.scalar(
            "PEAK_HIST/l1" + name_suffix, md['l1_peak_hist'], collections=[collection])
        tf.summary.scalar(
            "PEAK_HIST/log_l1" + name_suffix,
            md['log_l1_peak_hist'],
            collections=[collection])

        md['wasserstein_mass_hist'] = tf.placeholder(
            tf.float32, name='wasserstein_mass_hist' + name_suffix)
        md['l2_mass_hist'] = tf.placeholder(
            tf.float32, name='l2_mass_hist' + name_suffix)
        md['log_l2_mass_hist'] = tf.placeholder(
            tf.float32, name='log_l2_mass_hist' + name_suffix)
        md['l1_mass_hist'] = tf.placeholder(
            tf.float32, name='l1_mass_hist' + name_suffix)
        md['log_l1_mass_hist'] = tf.placeholder(
            tf.float32, name='log_l1_mass_hist' + name_suffix)
        md['total_stats_error_l1'] = tf.placeholder(
            tf.float32, name='total_stats_error_l1' + name_suffix)
        md['total_stats_error_l2'] = tf.placeholder(
            tf.float32, name='total_stats_error_l2' + name_suffix)
        tf.summary.scalar(
            "MASS_HIST/l2" + name_suffix, md['l2_mass_hist'], collections=[collection])
        tf.summary.scalar(
            "MASS_HIST/log_l2" + name_suffix,
            md['log_l2_mass_hist'],
            collections=[collection])
        tf.summary.scalar(
            "MASS_HIST/l1" + name_suffix, md['l1_mass_hist'], collections=[collection])
        tf.summary.scalar(
            "MASS_HIST/log_l1" + name_suffix,
            md['log_l1_mass_hist'],
            collections=[collection])
        tf.summary.scalar(
            "MASS_HIST/wasserstein_mass_hist" + name_suffix,
            md['wasserstein_mass_hist'],
            collections=[collection])
        tf.summary.scalar(
            "Combined_Stats/total_stats_error_l1" + name_suffix,
            md['total_stats_error_l1'],
            collections=[collection])
        tf.summary.scalar(
            "Combined_Stats/total_stats_error_l2" + name_suffix,
            md['total_stats_error_l2'],
            collections=[collection])

        plots = dict()
        plots['psd'] = PlotSummaryLog('Power_spectrum_density' + name_suffix, 'PLOT')
        plots['mass_hist'] = PlotSummaryLog('Mass_histogram' + name_suffix, 'PLOT')
        plots['peak_hist'] = PlotSummaryLog('Peak_histogram' + name_suffix, 'PLOT')

        return md, plots

    def _compute_real_stats(self, real):
        """Compute the main statistics on the real data."""
        stats = dict()

        psd_real, psd_axis = metrics.power_spectrum_batch_phys(
            X1=real, is_3d=self.is_3d)
        stats['psd_real'] = np.mean(psd_real, axis=0)
        stats['psd_axis'] = psd_axis
        del psd_real

        peak_hist_real, x_peak, lim_peak = metrics.peak_count_hist(dat=real)
        stats['peak_hist_real'] = peak_hist_real
        stats['x_peak'] = x_peak
        stats['lim_peak'] = lim_peak

        mass_hist_real, _, lim_mass = metrics.mass_hist(dat=real)
        lim_mass = list(lim_mass)
        lim_mass[1] = lim_mass[1]+1
        mass_hist_real, x_mass, lim_mass = metrics.mass_hist(dat=real, lim=lim_mass)
        stats['mass_hist_real'] = mass_hist_real
        stats['x_mass'] = x_mass
        stats['lim_mass'] = lim_mass

        del real
        return stats

    def train(self, dataset, resume=False):        
        if resume:
            self._stats = self.params['cosmology']['stats']
        else:
            real = self._backward_map(dataset.get_all_data())
            # This line should be improved, probably going to mess with Jonathan code
            if self.is_3d:
                if len(real.shape) > 4:
                    real = real[:, :, :, :, 0]
            else:
                if len(real.shape) > 3:
                    real = np.transpose(real, [0,3,1,2])
                    real = np.vstack(real)
            self._stats = self._compute_real_stats(real)
            self.params['cosmology']['stats'] = self._stats
        # Out of the _compute_real_stats function since we may want to change
        # this parameter during training.
        self._stats['N'] = self.params['cosmology']['Nstats']
        self._sum_data_iterator = itertools.cycle(dataset.iter(self._stats['N']))

        super().train(dataset=dataset, resume=resume)

    def _process_stat_dict(self, real, fake, _stats=None, _plots=None):
        """Calculates summary statistics based on given real and fake data"""
        if _stats is None:
            _stats = self._stats

        if _plots is None:
            _plots = self._plots

        stat_dict = dict()

        psd_gen, _ = metrics.power_spectrum_batch_phys(
            X1=fake, is_3d=self.is_3d)
        psd_gen = np.mean(psd_gen, axis=0)
        l2psd, logel2psd, l1psd, logel1psd = metrics.diff_vec(
            _stats['psd_real'], psd_gen)

        stat_dict['l2_psd'] = l2psd
        stat_dict['log_l2_psd'] = logel2psd
        stat_dict['l1_psd'] = l1psd
        stat_dict['log_l1_psd'] = logel1psd

        summary_str = _plots['psd'].produceSummaryToWrite(
            self._sess,
            _stats['psd_axis'],
            _stats['psd_real'],
            psd_gen)
        self._summary_writer.add_summary(summary_str, self._counter)

        peak_hist_fake, _, _ = metrics.peak_count_hist(
            fake, lim=_stats['lim_peak'])
        l2, logel2, l1, logel1 = metrics.diff_vec(
            _stats['peak_hist_real'], peak_hist_fake)

        stat_dict['l2_peak_hist'] = l2
        stat_dict['log_l2_peak_hist'] = logel2
        stat_dict['log_l1_peak_hist'] = logel1
        stat_dict['l1_peak_hist'] = l1

        summary_str = _plots['peak_hist'].produceSummaryToWrite(
            self._sess,
            _stats['x_peak'],
            _stats['peak_hist_real'],
            peak_hist_fake)

        self._summary_writer.add_summary(summary_str, self._counter)

        mass_hist_fake, _, _ = metrics.mass_hist(
            fake, lim=_stats['lim_mass'])
        l2, logel2, l1, logel1 = metrics.diff_vec(
            _stats['mass_hist_real'], mass_hist_fake)

        ws_hist = metrics.wasserstein_distance(
            _stats['mass_hist_real'],
            mass_hist_fake,
            safe=False)

        stat_dict['wasserstein_mass_hist'] = ws_hist
        stat_dict['l2_mass_hist'] = l2
        stat_dict['log_l2_mass_hist'] = logel2
        stat_dict['l1_mass_hist'] = l1
        stat_dict['log_l1_mass_hist'] = logel1

        summary_str = _plots['mass_hist'].produceSummaryToWrite(
            self._sess,
            _stats['x_mass'],
            _stats['mass_hist_real'],
            mass_hist_fake)
        self._summary_writer.add_summary(summary_str, self._counter)

        # Descriptive Stats
        descr_fake = np.array([metrics.describe(x) for x in fake])
        descr_real = np.array([metrics.describe(x) for x in real])

        stat_dict['descriptives'] = np.stack((np.mean(
            descr_fake, axis=0), np.mean(descr_real, axis=0)))

        # Distance of Peak Histogram
        index = np.random.choice(
            real.shape[0], min(50, real.shape[0]), replace=False
        )  # computing all pairwise comparisons is expensive
        peak_fake = np.array([
            metrics.peak_count(x, neighborhood_size=5, threshold=0)
            for x in fake[index]
        ])
        peak_real = np.array([
            metrics.peak_count(x, neighborhood_size=5, threshold=0)
            for x in real[index]
        ])

        # if tensorboard:
        stat_dict['peak_fake'] = np.log(np.hstack(peak_fake))
        stat_dict['peak_real'] = np.log(np.hstack(peak_real))
        stat_dict['distance_peak_comp'] = metrics.distance_chi2_peaks(
            peak_fake, peak_real)
        stat_dict['distance_peak_fake'] = metrics.distance_chi2_peaks(
            peak_fake, peak_fake)
        stat_dict['distance_peak_real'] = metrics.distance_chi2_peaks(
            peak_real, peak_real)

        # del peak_real, peak_fake

        # Measure Cross PS
        box_l = box_l = 100 / 0.7
        cross_rf, _ = metrics.power_spectrum_batch_phys(
            X1=real[index],
            X2=fake[index],
            box_l=box_l,
            is_3d=self.is_3d)
        cross_ff, _ = metrics.power_spectrum_batch_phys(
            X1=fake[index],
            X2=fake[index],
            box_l=box_l,
            is_3d=self.is_3d)
        cross_rr, _ = metrics.power_spectrum_batch_phys(
            X1=real[index],
            X2=real[index],
            box_l=box_l,
            is_3d=self.is_3d)

        stat_dict['cross_ps'] = [
            np.mean(cross_rf),
            np.mean(cross_ff),
            np.mean(cross_rr)
        ]

        stat_dict['total_stats_error_l1'] = metrics.total_stats_error(stat_dict, params=[1,0])
        stat_dict['total_stats_error_l2'] = metrics.total_stats_error(stat_dict, params=[0,1])

        return stat_dict

    def _process_stat_results(self, stat_dict):
        print(" [*] [Fake, Real] Min [{:.3f}, {:.3f}],\t"
              "Median [{:.3f},{:.3f}],\t"
              "Mean [{:.3E},{:.3E}],\t"
              "Max [{:.3E},{:.3E}],\t"
              "Var [{:.3E},{:.3E}]".format(
                stat_dict['descriptives'][0, 2],
                stat_dict['descriptives'][1, 2],
                stat_dict['descriptives'][0, 4],
                stat_dict['descriptives'][1, 4],
                stat_dict['descriptives'][0, 0],
                stat_dict['descriptives'][1, 0],
                stat_dict['descriptives'][0, 3],
                stat_dict['descriptives'][1, 3],
                stat_dict['descriptives'][0, 1],
                stat_dict['descriptives'][1, 1]))

        print(
            " [*] [Comp, Fake, Real] PeakDistance:[{:.3f}, {:.3f}, {:.3f}]"
            "\tCrossPS:[{:.3f}, {:.3f}, {:.3f}]".format(
                stat_dict['distance_peak_comp'],
                stat_dict['distance_peak_fake'],
                stat_dict['distance_peak_real'],
                stat_dict['cross_ps'][0],
                stat_dict['cross_ps'][1],
                stat_dict['cross_ps'][2]))
        # Save a summary if a new minimum of PSD is achieved
        l2_psd = stat_dict['l2_psd']
        if l2_psd < self._stats.get('best_psd',math.inf):
            print(' [*] New PSD Low achieved {:3f} (was {:3f})'.format(
                l2_psd, self._stats.get('best_psd', math.inf)))
            self._stats['best_psd'] = l2_psd
            self._save_current_step = True

        log_l2_psd = stat_dict['log_l2_psd']
        if log_l2_psd < self._stats.get('best_log_psd', math.inf):
            print(
                ' [*] New Log PSD Low achieved {:3f} (was {:3f})'.format(
                    log_l2_psd, self._stats.get('best_log_psd', math.inf)))
            self._stats['best_log_psd'] = log_l2_psd
            self._save_current_step = True
        print(' {} current PSD L2 {}, logL2 {}'.format(
            self._counter, stat_dict['l2_psd'], log_l2_psd))

        total_stats_error = stat_dict['total_stats_error_l2']
        if total_stats_error < self._stats.get('total_stats_error_l2', math.inf):
            print(
                ' [*] New l2 stats Low achieved {:3f} (was {:3f})'.format(
                    total_stats_error, self._stats.get('total_stats_error_l2', math.inf)))
            self._stats['total_stats_error_l2'] = total_stats_error
            self._save_current_step = True
        print(' {} current PSD L2 {}, logL2 {}, total {}'.format(
            self._counter, l2_psd, log_l2_psd, total_stats_error))

    def _train_log(self, feed_dict):
        super()._train_log(feed_dict)

        if np.mod(self._counter, self.params['sum_every']) == 0:
            z_sel = self._sample_latent(self._stats['N'])
            Xsel = next(self._sum_data_iterator)

            # reshape input according to 2d, 3d, or patch case
            Xsel = self.add_input_channel(Xsel)

            fake_image = self._generate_sample_safe(z_sel, Xsel)

            # pick only 1 channel for the patch case - the channel in which the original information is stored
            if self._is_3d:
                Xsel = Xsel[:, :, :, :, 0]
            else:
                if self.average_over_all_channels:
                    Xsel = np.transpose(Xsel, [0, 3, 1, 2])
                    Xsel = np.vstack(Xsel)
                else:
                    Xsel= Xsel[:, :, :, 0]

            real = self._backward_map(Xsel)

            fake = self._backward_map(fake_image)
            if not self._is_3d and len(fake.shape) == 4:
                if self.average_over_all_channels:
                    fake = np.transpose(fake, [0, 3, 1, 2])
                    fake = np.vstack(fake)
                else:
                    fake = fake[:, :, :, 0]
            fake = np.squeeze(fake)

            stat_dict = self._process_stat_dict(real, fake)
            for key in stat_dict.keys():
                feed_dict[self._md[key]] = stat_dict[key]

            summary_str = self._sess.run(
                self.summary_op_metrics, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_str, self._counter)

            # Print the results
            self._process_stat_results(stat_dict)

            # To save the stats in params
            self.params['cosmology']['stats'] = self._stats

    def generate(self,
                 N=None,
                 z=None,
                 X=None,
                 sess=None,
                 checkpoint=None,
                 **kwargs):
        '''
        Generate samples from already trained model.
        Arguments:
        Pass either N or z.
        Pass either sess or checkpoint.
        If sess is passed, it is assumed that the model has already been loaded using gan.load method
        '''
        if N and z:
            ValueError('Please choose between N and z')

        if sess and checkpoint:
            ValueError('Please choose between sess and checkpoint.\nIf sess is passed, it is assumed that the model is already loaded')

        images = super().generate(
            N=N, z=z, X=X, sess=sess, checkpoint=checkpoint, **kwargs)

        raw_images = self._backward_map(images)

        return images, raw_images


class TimeGAN(GAN):
    def __init__(self, params, model=None, is_3d=False):
        self.params = default_params_time(params)
        super().__init__(params=self.params, model=model, is_3d=is_3d)

    def _build_image_summary(self):
        vmin = tf.reduce_min(self._X)
        vmax = tf.reduce_max(self._X)
        for c in range(self.params["time"]["num_classes"]):
            tf.summary.image(
                "training/Real_Image_t{}".format(self.params['time']['classes'][c]),
                colorize(self._X[:, :, :, c:(c+1)], vmin, vmax),
                max_outputs=4,
                collections=['Images'])
            tf.summary.image(
                "training/Fake_Image_t{}".format(self.params['time']['classes'][c]),
                colorize(self._G_fake[:, :, :, c:(c+1)], vmin, vmax),
                max_outputs=4,
                collections=['Images'])

    def _sample_latent(self, bs=None):
        if bs is None:
            bs = self.batch_size
        latent = super()._sample_latent(bs=bs)
        return np.repeat(latent, self.num_classes, axis=0)

    def _sample_single_latent(self, bs=None):
        if bs is None:
            bs = 1
        return super(TimeGAN, self)._sample_latent(bs)

    def _get_sample_args(self, **kwargs):
        if "single_channel" in kwargs.keys():
            return self._model.G_c_fake
        return self._G_fake

    @property
    def num_classes(self):
        return self.params['time']['num_classes']


class TimeCosmoGAN(CosmoGAN, TimeGAN):
    def __init__(self, params, model=None, is_3d=False):
        super().__init__(params=params, model=model, is_3d=is_3d)
        self._md_t = dict()
        for t in range(self.num_classes):
            suff = '_t' + str(params['time']['classes'][t])
            self._md_t[t], self._plots[t] = CosmoGAN._init_logs('Time Cosmo Metrics', name_suffix=suff)
        self.summary_op_metrics_t = tf.summary.merge(
            tf.get_collection("Time Cosmo Metrics"))

    def train(self, dataset, resume=False):
        real = self._backward_map(dataset.get_all_data())
        self._stats_t = []
        for t in range(self.num_classes):
            self._stats_t.append(self._compute_real_stats(real[:,:,:,t]))
        super().train(dataset=dataset, resume=resume)

    def _train_log(self, feed_dict):
        super()._train_log(feed_dict)
        if np.mod(self._counter, self.params['sum_every']) == 0:
            z_sel = self._sample_latent(self._stats['N'])
            Xsel = next(self._sum_data_iterator)
            # reshape input according to 2d, 3d, or patch case
            Xsel = self.add_input_channel(Xsel)
            real_image = self._backward_map(Xsel)

            fake_image = self._generate_sample_safe(z_sel, Xsel)
            fake_image = self._backward_map(fake_image)

            for t in range(self.num_classes):
                real = real_image[:,:,:,t]
                fake = fake_image[:,:,:,t]

                stat_dict = self._process_stat_dict(real, fake, self._stats_t[t],
                                                    self._plots[t])
                for key in stat_dict.keys():
                    feed_dict[self._md_t[t][key]] = stat_dict[key]

            summary_str = self._sess.run(
                self.summary_op_metrics_t, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_str, self._counter)

    def _sample_latent(self, bs=None):
        return TimeGAN._sample_latent(self, bs)

    @property
    def average_over_all_channels(self):
        return True

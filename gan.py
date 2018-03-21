import tensorflow as tf
from tensorflow.contrib.data import Iterator
import numpy as np
import time
import os
import sys
import pickle
import scipy.misc
import scipy.ndimage.filters as filters
import utils
import metrics
from scipy import ndimage

from plot_summary import PlotSummaryLog
from default import default_params, default_params_cosmology
import data


class GAN(object):
    def __init__(self, params, model=None, is_3d=False):

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
        self._X = tf.placeholder(
            tf.float32, shape=[None, *self.params['image_size'], 1], name='X')

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

        if len(e_vars) and (
                'enc_learning_rate' in params['optimization'].keys()):
            self._has_encoder = True
        else:
            self._has_encoder = False

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # If input to be taken from files, create a dataset object, which will be initialized in train().
        # This dataset is intialized only once for the whole lifetime of the object
        if self.params['file_input']:
            self.dataset = None

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
        if self.is_3d:
            x_dim, y_dim, z_dim = self.params['image_size']
            num_images_in_each_row = utils.num_images_each_row(x_dim)

            self.real_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[
                    1, y_dim * (x_dim//num_images_in_each_row), z_dim * num_images_in_each_row, 1
                ])
            self.fake_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[
                    1, y_dim * (x_dim//num_images_in_each_row), z_dim * num_images_in_each_row, 1
                ])

            self.summary_op_real_image = tf.summary.image(
                "training/plot_real", self.real_placeholder)
            self.summary_op_fake_image = tf.summary.image(
                "training/plot_fake", self.fake_placeholder)

            if self.normalized():
                tf.summary.image(
                    "training/Real_Image_normalized",
                    self._normalize(self._X[:, 1, :, :, :]),
                    max_outputs=4,
                    collections=['Images'])
                tf.summary.image(
                    "training/Fake_Image_normalized",
                    self._normalize(self._G_fake[:, 1, :, :, :]),
                    max_outputs=4,
                    collections=['Images'])

        else:
            tf.summary.image(
                "training/Real_Image",
                self._X,
                max_outputs=4,
                collections=['Images'])
            tf.summary.image(
                "training/Fake_Image",
                self._G_fake,
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

        tf.summary.histogram('Prior/z', self._z, collections=['Images'])

        self.summary_op = tf.summary.merge(tf.get_collection("Training"))
        self.summary_op_img = tf.summary.merge(tf.get_collection("Images"))

        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

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

    def _buid_opt_summaries(self, optimizer_D, grads_and_vars_d, optimizer_G, grads_and_vars_g, optimizer_E):

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
            optim_learning_rate_G = tf.log(
                get_lr_ADAM(optimizer_G, gen_learning_rate))
            tf.summary.scalar(
                'Gen/Log_of_ADAM_learning_rate',
                optim_learning_rate_G,
                collections=["Training"])

            if optimizer_E is not None:
                optim_learning_rate_E = tf.log(
                    get_lr_ADAM(optimizer_E, enc_learning_rate))
                tf.summary.scalar(
                    'Gen/Log_of_ADAM_learning_rate',
                    optim_learning_rate_E,
                    collections=["Training"])

        if disc_optimizer == "adam":
            optim_learning_rate_D = tf.log(
                get_lr_ADAM(optimizer_D, disc_learning_rate))
            tf.summary.scalar(
                'Disc/Log_of_ADAM_learning_rate',
                optim_learning_rate_D,
                collections=["Training"])

    def train(self, X=None, resume=False):
        if self.params['file_input']: # samples to be read from file, rather than feeding on fly as numpy array
            if self.dataset is None:
                samples_dir_paths = self.params['samples_dir_paths']
                self.dataset, self.num_samples = data.create_input_pipeline(dir_paths=samples_dir_paths, 
                                                batch_size=self.batch_size, 
                                                k=self.params['cosmology']['k'])
                # create a reinitializable iterator given the dataset structure
                self.iterator = self.dataset.make_initializable_iterator()
                self.next_batch = self.iterator.get_next()
                # Op for initializing the iterator
                self.training_init_op = self.iterator.initializer

            n_data = self.num_samples

        else:
            n_data = len(X)

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
                self._load()
            else:
                self._sess.run(tf.global_variables_initializer())
                utils.saferm(self.params['summary_dir'])
                utils.saferm(self.params['save_dir'])
            if self.normalized():
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
                    if self.params['file_input']:
                        # Initialize iterator with the training dataset
                        self._sess.run(self.training_init_op)

                    idx = 0
                    while idx < self._n_batch:
                        if resume:
                            epoch = self.params['curr_epochs']
                            idx = self.params['curr_idx']
                            self._counter = self.params['curr_counter']
                            resume = False
                        else:
                            self.params['curr_epochs'] = epoch
                            self.params['curr_idx'] = idx
                            self.params['curr_counter'] = self._counter


                        if self.params['file_input']:
                            # Initialize iterator with the training dataset
                            X_real = self._sess.run(self.next_batch)
                        else:
                            X_real = X[idx * self.batch_size:(idx + 1) * self.batch_size]
                            
                        X_real.resize([*X_real.shape, 1])
                        
                        for _ in range(5):
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
                            print(
                                "Epoch: [{:2d}] [{:4d}/{:4d}] "
                                "Counter:{:2d}\t"
                                "({:4.1f} min\t"
                                "{:4.3f} {:4.2f} examples/sec\t"
                                "sec/batch)\t"
                                "L_Disc:{:.8f}\t"
                                "L_Gen:{:.8f}".
                                format(epoch, idx, self._n_batch,
                                       self._counter,
                                       (current_time - start_time) / 60,
                                       100.0 * self.batch_size /
                                       (current_time - prev_iter_time),
                                       (current_time - prev_iter_time) / 100,
                                       loss_d, loss_g))
                            prev_iter_time = current_time

                        self._train_log(self._get_dict(sample_z, X_real), X, epoch=epoch, batch_num=idx)

                        if (np.mod(self._counter, self.params['save_every'])
                                == 0) | self._save_current_step:
                            self._save(self._savedir, self._counter)
                            self._save_current_step = False
                        self._counter += 1
                        idx += 1
                    epoch += 1
            except KeyboardInterrupt:
                pass
            self._save(self._savedir, self._counter)

    def _train_log(self, feed_dict, X, epoch=None, batch_num=None):
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
                        utils.tile_cube_slices(real_arr[0, :, :, :, 0], str(epoch), str(batch_num), 'real', True),
                        self.fake_placeholder:
                        utils.tile_cube_slices(fake_arr[0, :, :, :, 0], str(epoch), str(batch_num), 'fake', True)
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
        if self.params['num_classes'] > 1:
            latent = utils.sample_latent(
                int(np.ceil(bs / self.params['num_classes'])), latent_dim,
                self._prior_distribution)
            return np.repeat(latent, self.params['num_classes'], axis=0)[:bs]
        return utils.sample_latent(bs, latent_dim, self._prior_distribution)

    def _get_dict(self, z=None, X=None, y=None, index=None):
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
        if y is not None:
            if index is not None:
                feed_dict[self._model.y] = y[index]
            else:
                feed_dict[self._model.y] = y

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

        with open(self.params['save_dir'] + 'params.pkl', 'wb') as f:
            pickle.dump(self.params, f)

    def _load(self, file_name=None):
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
                 y=None,
                 sess=None,
                 file_name=None):
        if N and z:
            ValueError('Please choose between N and z')
        if sess is not None:
            self._sess = sess
            return self._generate_sample(
                N=N, z=z, X=X, y=y, file_name=file_name)
        with tf.Session() as self._sess:
            return self._generate_sample(
                N=N, z=z, X=X, y=y, file_name=file_name)

    def _generate_sample(self, N=None, z=None, X=None, y=None,
                         file_name=None):
        self._load(file_name=file_name)

        if z is None:
            if N is None:
                N = self.batch_size
        z = self._sample_latent(N)
        return self._generate_sample_safe(z, X, y)

    def _get_sample_args(self):
        return self._G_fake

    def _special_vstack(self, gi):
        if type(gi[0]) is np.ndarray:
            return np.vstack(gi)
        else:
            s = []
            for j in range(len(gi[0])):
                s.append(np.vstack([el[j] for el in gi]))
            return tuple(s)

    def _generate_sample_safe(self, z=None, X=None, y=None):
        gen_images = []
        N = len(z)
        sind = 0
        bs = self.batch_size
        if N > bs:
            nb = (N - 1) // bs
            for i in range(nb):
                gi = self._sess.run(
                    self._get_sample_args(),
                    feed_dict=self._get_dict(z, X, y, slice(sind, sind + bs)))
                gen_images.append(gi)
                sind = sind + bs
        gi = self._sess.run(
            self._get_sample_args(),
            feed_dict=self._get_dict(z, X, y, slice(sind, N)))
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
        return self._has_encoder

    @property
    def model_name(self):
        return self._model_name

    @property
    def is_3d(self):
        return self._is_3d


class CosmoGAN(GAN):
    def __init__(self, params, model=None, is_3d=False):
        super().__init__(params=params, model=model, is_3d=is_3d)

        self.params = default_params_cosmology(self.params)
        assert (self.params['cosmology']['k'] > 0)
        self._G_raw = utils.inv_pre_process(self._G_fake,
                                            self.params['cosmology']['k'])
        self._X_raw = utils.inv_pre_process(self._X,
                                            self.params['cosmology']['k'])

        self._md = dict()

        self._md['descriptives'] = tf.placeholder(
            tf.float64, shape=[2, 5], name="DescriptiveStatistics")

        tf.summary.scalar(
            "descriptives/mean_Fake",
            self._md['descriptives'][0, 0],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/var_Fake",
            self._md['descriptives'][0, 1],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/min_Fake",
            self._md['descriptives'][0, 2],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/max_Fake",
            self._md['descriptives'][0, 3],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/median_Fake",
            self._md['descriptives'][0, 4],
            collections=['Metrics'])

        tf.summary.scalar(
            "descriptives/mean_Real",
            self._md['descriptives'][1, 0],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/var_Real",
            self._md['descriptives'][1, 1],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/min_Real",
            self._md['descriptives'][1, 2],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/max_Real",
            self._md['descriptives'][1, 3],
            collections=['Metrics'])
        tf.summary.scalar(
            "descriptives/median_Real",
            self._md['descriptives'][1, 4],
            collections=['Metrics'])

        self._md['peak_fake'] = tf.placeholder(
            tf.float64, shape=[None], name="peak_fake")
        self._md['peak_real'] = tf.placeholder(
            tf.float64, shape=[None], name="peak_real")
        tf.summary.histogram(
            "Peaks/Fake_log", self._md['peak_fake'], collections=['Metrics'])
        tf.summary.histogram(
            "Peaks/Real_log", self._md['peak_real'], collections=['Metrics'])

        self._md['distance_peak_comp'] = tf.placeholder(
            tf.float64, name='distance_peak_comp')
        self._md['distance_peak_fake'] = tf.placeholder(
            tf.float64, name='distance_peak_fake')
        self._md['distance_peak_real'] = tf.placeholder(
            tf.float64, name='distance_peak_real')

        tf.summary.scalar(
            "Peaks/Ch2_Fake-Real",
            self._md['distance_peak_comp'],
            collections=['Metrics'])
        tf.summary.scalar(
            "Peaks/Ch2_Fake-Fake",
            self._md['distance_peak_fake'],
            collections=['Metrics'])
        tf.summary.scalar(
            "Peaks/Ch2_Real-Real",
            self._md['distance_peak_real'],
            collections=['Metrics'])

        self._md['cross_ps'] = tf.placeholder(
            tf.float64, shape=[3], name='cross_ps')

        tf.summary.scalar(
            "PSD/Cross_Fake-Real",
            self._md['cross_ps'][0],
            collections=['Metrics'])
        tf.summary.scalar(
            "PSD/Cross_Fake-Fake",
            self._md['cross_ps'][1],
            collections=['Metrics'])
        tf.summary.scalar(
            "PSD/Cross_Real-Real",
            self._md['cross_ps'][2],
            collections=['Metrics'])

        tf.summary.histogram(
            "Pixel/Fake", self._G_fake, collections=['Metrics'])
        tf.summary.histogram("Pixel/Real", self._X, collections=['Metrics'])

        # This summary needs clipping
        clip_max = 1e10
        tf.summary.histogram(
            "Pixel/Fake_Raw",
            tf.clip_by_value(self._G_raw, 0, clip_max),
            collections=['Metrics'])
        tf.summary.histogram(
            "Pixel/Real_Raw",
            tf.clip_by_value(self._X_raw, 0, clip_max),
            collections=['Metrics'])

        self._md['l2_psd'] = tf.placeholder(tf.float32, name='l2_psd')
        self._md['log_l2_psd'] = tf.placeholder(tf.float32, name='log_l2_psd')
        self._md['l1_psd'] = tf.placeholder(tf.float32, name='l1_psd')
        self._md['log_l1_psd'] = tf.placeholder(tf.float32, name='log_l1_psd')
        tf.summary.scalar(
            "PSD/l2", self._md['l2_psd'], collections=['Metrics'])
        tf.summary.scalar(
            "PSD/log_l2", self._md['log_l2_psd'], collections=['Metrics'])
        tf.summary.scalar(
            "PSD/l1", self._md['l1_psd'], collections=['Metrics'])
        tf.summary.scalar(
            "PSD/log_l1", self._md['log_l1_psd'], collections=['Metrics'])

        if self.params['num_classes'] > 1:
            for i in range(self.params['num_classes']):
                self._md['c' + str(i) + '_l2_psd'] = tf.placeholder(tf.float32, name='c' + str(i) + '_l2_psd')
                self._md['c' + str(i) + '_log_l2_psd'] = tf.placeholder(tf.float32, name='c' + str(i) + '_log_l2_psd')
                self._md['c' + str(i) + '_l1_psd'] = tf.placeholder(tf.float32, name='c' + str(i) + '_l1_psd')
                self._md['c' + str(i) + '_log_l1_psd'] = tf.placeholder(tf.float32, name='c' + str(i) + '_log_l1_psd')
                tf.summary.scalar(
                    "PSD/l2", self._md['c' + str(i) + '_l2_psd'], collections=['Metrics'])
                tf.summary.scalar(
                    "PSD/log_l2", self._md['c' + str(i) + '_log_l2_psd'], collections=['Metrics'])
                tf.summary.scalar(
                    "PSD/l1", self._md['c' + str(i) + '_l1_psd'], collections=['Metrics'])
                tf.summary.scalar(
                    "PSD/log_l1", self._md['c' + str(i) + '_log_l1_psd'], collections=['Metrics'])

        self._md['l2_peak_hist'] = tf.placeholder(
            tf.float32, name='l2_peak_hist')
        self._md['log_l2_peak_hist'] = tf.placeholder(
            tf.float32, name='log_l2_peak_hist')
        self._md['l1_peak_hist'] = tf.placeholder(
            tf.float32, name='l1_peak_hist')
        self._md['log_l1_peak_hist'] = tf.placeholder(
            tf.float32, name='log_l1_peak_hist')
        tf.summary.scalar(
            "PEAK_HIST/l2", self._md['l2_peak_hist'], collections=['Metrics'])
        tf.summary.scalar(
            "PEAK_HIST/log_l2",
            self._md['log_l2_peak_hist'],
            collections=['Metrics'])
        tf.summary.scalar(
            "PEAK_HIST/l1", self._md['l1_peak_hist'], collections=['Metrics'])
        tf.summary.scalar(
            "PEAK_HIST/log_l1",
            self._md['log_l1_peak_hist'],
            collections=['Metrics'])

        self._md['l2_mass_hist'] = tf.placeholder(
            tf.float32, name='l2_mass_hist')
        self._md['log_l2_mass_hist'] = tf.placeholder(
            tf.float32, name='log_l2_mass_hist')
        self._md['l1_mass_hist'] = tf.placeholder(
            tf.float32, name='l1_mass_hist')
        self._md['log_l1_mass_hist'] = tf.placeholder(
            tf.float32, name='log_l1_mass_hist')
        tf.summary.scalar(
            "MASS_HIST/l2", self._md['l2_mass_hist'], collections=['Metrics'])
        tf.summary.scalar(
            "MASS_HIST/log_l2",
            self._md['log_l2_mass_hist'],
            collections=['Metrics'])
        tf.summary.scalar(
            "MASS_HIST/l1", self._md['l1_mass_hist'], collections=['Metrics'])
        tf.summary.scalar(
            "MASS_HIST/log_l1",
            self._md['log_l1_mass_hist'],
            collections=['Metrics'])

        self._psd_plot = PlotSummaryLog('Power_spectrum_density', 'PLOT')
        self._mass_hist_plot = PlotSummaryLog('Mass_histogram', 'PLOT')
        self._peak_hist_plot = PlotSummaryLog('Peak_histogram', 'PLOT')

        if self.params['num_classes'] > 1:
            self._c_psd_plot = []
            for i in range(self.params['num_classes']):
                self._c_psd_plot.append(PlotSummaryLog('C' + str(i) + '_Power_spectrum_density', 'PLOT'))

        self.summary_op_metrics = tf.summary.merge(
            tf.get_collection("Metrics"))

    def _compute_real_psd(self, X=None):
        '''
        Compute the real PSD on 'max_num_psd' data
        '''
        self._Npsd = self.params['cosmology']['Npsd']
        self._max_num_psd = self.params['cosmology']['max_num_psd']

        if self.params['file_input']: # training samples stored in files
            #X, _ = data.load_3d_hists(self.params['samples_dir_path'], self.params['cosmology']['k'])
            psd_real = metrics.power_spectrum_batch_phys_from_file_input(self.params['samples_dir_paths'],
                                        self.params['image_size'],
                                        self.params['cosmology']['k'], 
                                        self._max_num_psd,
                                        is_3d=self.is_3d)
            self._psd_real = np.mean(psd_real, axis=0)

        else: # training samples passed as numpy ndarray
            psd_real, _ = metrics.power_spectrum_batch_phys(
                X1=utils.backward_map(X, self.params['cosmology']['k']), 
                                        is_3d=self.is_3d)
            self._psd_real = np.mean(psd_real, axis=0)
            del psd_real
        
        if self.params['num_classes'] > 1: # this block will not work with file inputs!!
            self._c_psd_real = []
            for i in range(self.params['num_classes']):
                psd_real, _ = metrics.power_spectrum_batch_phys(
                    X1=utils.backward_map(X[i::self.params['num_classes']],
                                          self.params['cosmology']['k']),
                    is_3d=self.is_3d)
                self._c_psd_real.append(np.mean(psd_real, axis=0))
                del psd_real

    def train(self, X=None, resume=False):
        if X is not None and self.params['file_input']:
            print("Warning: A numpy array as well as a file location for training set is provided!!")

        self._compute_real_psd(X)

        if resume:
            self.best_psd = self.params['cosmology']['best_psd']
            self.best_log_psd = self.params['cosmology']['best_log_psd']
        else:
            self.best_psd = 1e10
            self.best_log_psd = 10000
            self.params['cosmology']['best_psd'] = self.best_psd
            self.params['cosmology']['best_log_psd'] = self.best_log_psd

        super().train(X=X, resume=resume)

    def _multiclass_l2_psd(self, feed_dict, X):
        Xsel = X[0:self._Npsd]
        real = utils.backward_map(Xsel, self.params['cosmology']['k'])
        z_sel = self._sample_latent(self._Npsd)

        _, fake = self._generate_sample_safe(
            z_sel, Xsel.reshape([self._Npsd, *X.shape[1:], 1]))
        fake.resize([self._Npsd, *X.shape[1:]])

        nc = self.params['num_classes']
        for i in range(nc):
            psd_gen, x = metrics.power_spectrum_batch_phys(
                X1=fake[i::nc], is_3d=self.is_3d)
            psd_gen = np.mean(psd_gen, axis=0)
            l2, logel2, l1, logel1 = metrics.diff_vec(self._c_psd_real[i], psd_gen)

            feed_dict[self._md['c' + str(i) + '_l2_psd']] = l2
            feed_dict[self._md['c' + str(i) + '_log_l2_psd']] = logel2
            feed_dict[self._md['c' + str(i) + '_l1_psd']] = l1
            feed_dict[self._md['c' + str(i) + '_log_l1_psd']] = logel1

            summary_str = self._c_psd_plot[i].produceSummaryToWrite(
                self._sess, x, self._c_psd_real[i], psd_gen)
            self._summary_writer.add_summary(summary_str, self._counter)

    def _sample_real_data(self, X):
        '''
        Sample from training data, where training data coould be both, a numpy array or files saved on disk
        '''
        if X is not None: # training set provided as numpy array
            ind = np.random.randint(0, len(X), size=[self._Npsd])
            Xsel = X[ind]
            real = utils.backward_map(Xsel, self.params['cosmology']['k'])
        else:        
            num_dir = len(self.params['samples_dir_paths'])
            num_samples_each_dir = [ self._Npsd//num_dir for i in range(num_dir)]
            num_samples_each_dir[0] += (self._Npsd % num_dir) # remainder samples sampled from 0th directory

            for i, dir_path in enumerate(self.params['samples_dir_paths']): # sample from each directory
                forward_mapped_data, _ = data.read_tfrecords_from_dir(dir_path, 
                                                    self.params['image_size'], 
                                                    self.params['cosmology']['k']) # load all the samples in the current directory

                ind = np.random.randint(0, len(forward_mapped_data), num_samples_each_dir[i]) 

                if i == 0:
                    Xsel = forward_mapped_data[ind] # sample only required number
                    real = utils.backward_map(Xsel, self.params['cosmology']['k'])
                else:
                    Xsel = np.vstack((Xsel, forward_mapped_data[ind] ))
                    real = utils.backward_map(Xsel, self.params['cosmology']['k'])

        return real, Xsel

    def _train_log(self, feed_dict, X, epoch=None, batch_num=None):
        super()._train_log(feed_dict, X, epoch, batch_num)

        if np.mod(self._counter, self.params['sum_every']) == 0:
            if self.params['num_classes'] > 1:
                self._multiclass_l2_psd(feed_dict, X)
            
            real, Xsel = self._sample_real_data(X)

            z_sel = self._sample_latent(self._Npsd)
            _, fake = self._generate_sample_safe(
                z_sel, Xsel.reshape([self._Npsd, *Xsel.shape[1:], 1]))

            fake.resize([self._Npsd, *Xsel.shape[1:]])

            psd_gen, x = metrics.power_spectrum_batch_phys(
                X1=fake, is_3d=self.is_3d)
            psd_gen = np.mean(psd_gen, axis=0)
            l2psd, logel2psd, l1psd, logel1psd = metrics.diff_vec(self._psd_real, psd_gen)

            feed_dict[self._md['l2_psd']] = l2psd
            feed_dict[self._md['log_l2_psd']] = logel2psd
            feed_dict[self._md['l1_psd']] = l1psd
            feed_dict[self._md['log_l1_psd']] = logel1psd

            summary_str = self._psd_plot.produceSummaryToWrite(
                self._sess, x, self._psd_real, psd_gen)
            self._summary_writer.add_summary(summary_str, self._counter)

            y_real, y_fake, x = metrics.peak_count_hist(real, fake)
            l2, logel2, l1, logel1 = metrics.diff_vec(y_real, y_fake)

            feed_dict[self._md['l2_peak_hist']] = l2
            feed_dict[self._md['log_l2_peak_hist']] = logel2
            feed_dict[self._md['log_l1_peak_hist']] = logel1
            feed_dict[self._md['l1_peak_hist']] = l1

            summary_str = self._peak_hist_plot.produceSummaryToWrite(
                self._sess, x, y_real, y_fake)
            self._summary_writer.add_summary(summary_str, self._counter)

            y_real, y_fake, x = metrics.mass_hist(real, fake)
            l2, logel2, l1, logel1 = metrics.diff_vec(y_real, y_fake)

            feed_dict[self._md['l2_mass_hist']] = l2
            feed_dict[self._md['log_l2_mass_hist']] = logel2
            feed_dict[self._md['l1_mass_hist']] = l1
            feed_dict[self._md['log_l1_mass_hist']] = logel1

            summary_str = self._mass_hist_plot.produceSummaryToWrite(
                self._sess, x, y_real, y_fake)
            self._summary_writer.add_summary(summary_str, self._counter)

            real = utils.makeit_square(real)
            fake = utils.makeit_square(fake)

            if self.params['cosmology']['clip_max_real']:
                clip_max = real.ravel().max()
            else:
                clip_max = 1e10

            fake = np.clip(
                np.nan_to_num(fake), self.params['cosmology']['log_clip'],
                clip_max)
            real = np.clip(
                np.nan_to_num(real), self.params['cosmology']['log_clip'],
                clip_max)

            if self.params['cosmology']['sigma_smooth'] is not None:
                fake = ndimage.gaussian_filter(
                    fake, sigma=self.params['cosmology']['sigma_smooth'])
                real = ndimage.gaussian_filter(
                    real, sigma=self.params['cosmology']['sigma_smooth'])

            # Descriptive Stats
            descr_fake = np.array([metrics.describe(x) for x in fake])
            descr_real = np.array([metrics.describe(x) for x in real])

            feed_dict[self._md['descriptives']] = np.stack((np.mean(
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
            feed_dict[self._md['peak_fake']] = np.log(np.hstack(peak_fake))
            feed_dict[self._md['peak_real']] = np.log(np.hstack(peak_real))
            feed_dict[self._md[
                'distance_peak_comp']] = metrics.distance_chi2_peaks(
                    peak_fake, peak_real)
            feed_dict[self._md[
                'distance_peak_fake']] = metrics.distance_chi2_peaks(
                    peak_fake, peak_fake)
            feed_dict[self._md[
                'distance_peak_real']] = metrics.distance_chi2_peaks(
                    peak_real, peak_real)

            # del peak_real, peak_fake

            # Measure Cross PS
            box_l = box_l = 100 / 0.7
            cross_rf, _ = metrics.power_spectrum_batch_phys(
                X1=real[index], X2=fake[index], box_l=box_l, is_3d=self.is_3d)
            cross_ff, _ = metrics.power_spectrum_batch_phys(
                X1=fake[index], X2=fake[index], box_l=box_l, is_3d=self.is_3d)
            cross_rr, _ = metrics.power_spectrum_batch_phys(
                X1=real[index], X2=real[index], box_l=box_l, is_3d=self.is_3d)

            feed_dict[self._md['cross_ps']] = [
                np.mean(cross_rf),
                np.mean(cross_ff),
                np.mean(cross_rr)
            ]

            # del cross_ff, cross_rf, cross_rr

            summary_str = self._sess.run(
                self.summary_op_metrics, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_str, self._counter)
            # Print the results
            print(
                " [*] [Fake, Real] Min [{:.3f}, {:.3f}],\t"
                "Median [{:.3f},{:.3f}],\t"
                "Mean [{:.3E},{:.3E}],\t"
                "Max [{:.3E},{:.3E}],\t"
                "Var [{:.3E},{:.3E}]".
                format(feed_dict[self._md['descriptives']][0, 2],
                       feed_dict[self._md['descriptives']][1, 2],
                       feed_dict[self._md['descriptives']][0, 4],
                       feed_dict[self._md['descriptives']][1, 4],
                       feed_dict[self._md['descriptives']][0, 0],
                       feed_dict[self._md['descriptives']][1, 0],
                       feed_dict[self._md['descriptives']][0, 3],
                       feed_dict[self._md['descriptives']][1, 3],
                       feed_dict[self._md['descriptives']][0, 1],
                       feed_dict[self._md['descriptives']][1, 1]))

            print(
                " [*] [Comp, Fake, Real] PeakDistance:[{:.3f}, {:.3f}, {:.3f}]"
                "\tCrossPS:[{:.3f}, {:.3f}, {:.3f}]".
                format(feed_dict[self._md['distance_peak_comp']],
                       feed_dict[self._md['distance_peak_fake']],
                       feed_dict[self._md['distance_peak_real']],
                       feed_dict[self._md['cross_ps']][0],
                       feed_dict[self._md['cross_ps']][1],
                       feed_dict[self._md['cross_ps']][2]))
            # Save a summary if a new minimum of PSD is achieved
            if l2psd < self.best_psd:
                print(' [*] New PSD Low achieved {:3f} (was {:3f})'.format(
                    l2psd, self.best_psd))
                self.best_psd, self._save_current_step = l2psd, True
            if logel2psd < self.best_log_psd:
                print(
                    ' [*] New Log PSD Low achieved {:3f} (was {:3f})'.format(
                        logel2psd, self.best_log_psd))
                self.best_log_psd, self._save_current_step = logel2psd, True
            print(' {} current PSD L2 {}, logL2 {}'.format(
                self._counter, l2psd, logel2psd))

    def _get_sample_args(self):
        return [self._G_fake, self._G_raw]

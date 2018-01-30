import tensorflow as tf
import numpy as np
import utils, metrics, optimization
import time
import os, sys
import pickle


class GAN(object):
    def __init__(self, params, model=None):

        tf.reset_default_graph()
        self.params = params
        if model is None:
            model = params['model']
        else:
            params['model'] = model
        self._savedir = params['save_dir']
        self._sess = None
        self.batch_size = self.params['optimization']['batch_size']
        
        self._mean = tf.get_variable(name="mean", dtype=tf.float32, shape=[1], trainable=False, initializer = tf.constant_initializer(0.))
        self._var = tf.get_variable(name="var", dtype=tf.float32, shape=[1], trainable=False, initializer = tf.constant_initializer(1.))

        self._z = tf.placeholder(tf.float32, shape=[None, self.params['generator']['latent_dim']], name='z')
        self._X = tf.placeholder(tf.float32, shape=[None, *self.params['image_size'], 1], name='X')

        name = params['name']
        self._model = model(params, self._normalize(self._X), self._z, name=name if name else None)
        self._model_name = self._model.name
        self._D_loss = self._model.D_loss
        self._G_loss = self._model.G_loss
        self._G_fake = self._unnormalize(self._model.G_fake)
        assert(self.params['k']>0)
        self._G_raw = utils.inv_pre_process(self._G_fake, self.params['k'])
        self._X_raw = utils.inv_pre_process(self._X, self.params['k'])

        t_vars = tf.trainable_variables()
        utils.show_all_variables()

        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        e_vars = [var for var in t_vars if 'encoder' in var.name]

        if len(e_vars) and ('enc_learning_rate' in params['optimization'].keys()):
            self._has_encoder = True
        else:
            self._has_encoder = False

        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer_D, optimizer_G, optimizer_E = optimization.build_optmizer(self.params['optimization'], self.has_encoder)

        grads_and_vars_d = optimizer_D.compute_gradients(self._D_loss, var_list=d_vars)
        grads_and_vars_g = optimizer_G.compute_gradients(self._G_loss, var_list=g_vars)

        self.D_solver = optimizer_D.apply_gradients(grads_and_vars_d, global_step=global_step)
        self.G_solver = optimizer_G.apply_gradients(grads_and_vars_g, global_step=global_step)

        if self.has_encoder:
            self.E_loss = self._model.E_loss
            grads_and_vars_e = optimizer_E.compute_gradients(self.E_loss, var_list=e_vars)
            self.E_solver = optimizer_E.apply_gradients(grads_and_vars_e, global_step=global_step)

        optimization.buid_opt_summaries(optimizer_D,
                                        grads_and_vars_d,
                                        optimizer_G,
                                        grads_and_vars_g,
                                        optimizer_E,
                                        self.params['optimization'])

        # Summaries
        tf.summary.image("trainingBW/Real_Image", self._X , max_outputs=4, collections=['Images'])
        tf.summary.image("trainingBW/Fake_Image", self._G_fake, max_outputs=4, collections=['Images'])
        tf.summary.histogram('Prior/z', self._z, collections=['Images'])

        # To improve...
        self.metr_dict = metrics.build_metrics_summaries(real=self._X, real_raw=self._X_raw, fake=self._G_fake, fake_raw=self._G_raw,
                                                    batch_size=self.batch_size)

        self.l2_psd = tf.placeholder(tf.float32, name = 'l2_psd')
        self.log_l2_psd = tf.placeholder(tf.float32, name = 'log_l2_psd')
        self.l1_psd = tf.placeholder(tf.float32, name = 'l1_psd')
        self.log_l1_psd = tf.placeholder(tf.float32, name = 'log_l1_psd')
        tf.summary.scalar("PSD/l2", self.l2_psd, collections=['metrics'])
        tf.summary.scalar("PSD/log_l2", self.log_l2_psd, collections=['metrics'])
        tf.summary.scalar("PSD/l1", self.l1_psd, collections=['metrics'])
        tf.summary.scalar("PSD/log_l1", self.log_l1_psd, collections=['metrics'])

        self.summary_op = tf.summary.merge(tf.get_collection("Training"))
        self.summary_op_img = tf.summary.merge(tf.get_collection("Images"))
        self.summary_op_metrics = tf.summary.merge(tf.get_collection("metrics"))
        
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        if 'prior_distribution' in self.params.keys():
            self._prior_distribution = self.params['prior_distribution']
        else:
            self._prior_distribution = 'uniform'

    def train(self, X):

        N = 500
        # Compute the real PSD on all data! May take some time
        psd_real, _ = metrics.power_spectrum_batch_phys(X1=utils.backward_map(X, self.params['k']))
        psd_real = np.mean(psd_real, axis=0)

        # Saving the objects:
        os.makedirs(self.params['save_dir'], exist_ok=True)
        with open(self.params['save_dir']+'params.pkl', 'wb') as f:
            pickle.dump(self.params, f)

        n_data  = len(X)
        self.counter = 1
        start_time = time.time()
        prev_iter_time = start_time

        self.n_epoch = self.params['optimization']['epoch']
        summary_dir = self.params['summary_dir'] 
        sum_every = self.params['sum_every']
        viz_every = self.params['viz_every']
        self.best_psd = 1e10
        self.best_log_psd = 10000
        # ngen = 100
        self.total_iter =  self.n_epoch * (n_data // self.batch_size) - 1
        self.n_batch = n_data // self.batch_size
        save_current_step = False

        utils.saferm(summary_dir)
        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as self._sess:
            self._sess.run(tf.global_variables_initializer())
            y_vec = self._get_classes(self.batch_size)
            if 'normalize' in self.params.keys() and self.params['normalize']:
                m = np.mean(X)
                v = np.var(X-m)
                self._mean.assign([m]).eval()
                self._var.assign([v]).eval()
            self._var.eval()
            self._mean.eval()
            summary_writer = tf.summary.FileWriter(summary_dir, self._sess.graph)

            for epoch in range(self.n_epoch):
                for idx in range(self.n_batch):
                    X_real = X[idx*self.batch_size:(idx+1)*self.batch_size]
                    X_real.resize([*X_real.shape,1])
                    for _ in range(5):
                        sample_z = self._sample_latent(self.batch_size)
                        _, loss_d = self._sess.run([self.D_solver, self._D_loss], feed_dict=self._get_dict(sample_z, X_real, y_vec))
                        if self.has_encoder:
                            _, loss_e = self._sess.run([self.E_solver, self.E_loss], feed_dict=self._get_dict(sample_z, X_real, y_vec))
                    sample_z = self._sample_latent(self.batch_size)
                    _, loss_g, v, m = self._sess.run([self.G_solver, self._G_loss, self._var, self._mean], feed_dict=self._get_dict(sample_z, X_real, y_vec))
                    if np.mod(self.counter, 100) == 0:
                        current_time = time.time()
                        print(
                            "Epoch: [{:2d}] [{:4d}/{:4d}] Counter:{:2d}\t({:4.1f} min\t{:4.2f} examples/sec\t{:4.3f} sec/batch)\tL_Disc:{:.8f}\tL_Gen:{:.8f}".format(
                                epoch, idx, self.n_batch, self.counter, (current_time - start_time) / 60,
                                                                 100.0 * self.batch_size / (current_time - prev_iter_time),
                                                                 (current_time - prev_iter_time) / 100, loss_d, loss_g))
                        prev_iter_time = current_time

                    if np.mod(self.counter, self.params['sum_every']) == 0:
                        summary_str = self._sess.run(self.summary_op, feed_dict=self._get_dict(sample_z, X_real, y_vec))
                        summary_writer.add_summary(summary_str, self.counter)

                    if np.mod(self.counter, self.params['viz_every']) == 0:
                        feed_dict = self._get_dict(sample_z, X_real, y_vec)

                        # BW Images
                        summary_str = self._sess.run(self.summary_op_img, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, self.counter)

                        # RAW Images for metrics and color images
                        fake, real = self._sess.run([self._G_raw, self._X_raw], feed_dict= feed_dict)

                        m = metrics.calculate_metrics(fake, real, self.params)
                        for key in self.metr_dict:
                            feed_dict[self.metr_dict[key]] = m[key]

                        sample_z_large = self._sample_latent(N)
                        _, fake_sample_large = self._generate_sample_safe(sample_z_large, X[:N].reshape([N,X.shape[1],X.shape[2],1]))
                        fake_sample_large.resize([N,X.shape[1],X.shape[2]])
                        psd_gen, _ = metrics.power_spectrum_batch_phys(X1=fake_sample_large)
                        psd_gen = np.mean(psd_gen, axis=0)

                        e = psd_real - psd_gen
                        l2 = np.mean(e*e)
                        l1 = np.mean(np.abs(e))
                        loge = 10*(np.log10(psd_real+1e-5) - np.log10(psd_gen+1e-5))
                        logel2 = np.mean(loge*loge)
                        logel1 = np.mean(np.abs(loge))
                        feed_dict[self.l2_psd] = l2
                        feed_dict[self.log_l2_psd] = logel2
                        feed_dict[self.l1_psd] = l1
                        feed_dict[self.log_l1_psd] = logel1

                        summary_str = self._sess.run(self.summary_op_metrics, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, self.counter)
                        # Save a summary if a new minimum of PSD is achieved
                        if l2 < self.best_psd:
                            print(' [*] New PSD Low achieved {:3f} (was {:3f})'.format(l2,self.best_psd))
                            self.best_psd, save_current_step = l2, True
                        if logel2 < self.best_log_psd:
                            print(' [*] New Log PSD Low achieved {:3f} (was {:3f})'.format(logel2,self.best_log_psd))
                            self.best_log_psd, save_current_step = logel2, True
                        print(' {} current PSD L2 {}, logL2 {}'.format(self.counter, l2, logel2))
                    if (np.mod(self.counter, self.params['save_every']) == 0) | (self.counter == self.total_iter) | save_current_step:
                        self._save(self._savedir, self.counter)
                        save_current_step = False
                    self.counter += 1

    def _sample_latent(self, bs=None):
        if bs is None:
            bs = self.batch_size
        latent_dim = self.params['generator']['latent_dim']
        return utils.sample_latent(bs, latent_dim, self._prior_distribution)

    def _get_classes(self, bs=None):
        if bs is None:
            bs = self.batch_size
        if 'num_classes' not in self.params or self.params['num_classes'] <= 1:
            return None
        return np.resize(np.arange(self.params['num_classes']), (bs, 1)) / (self.params['num_classes'] - 1.0)

    def _get_dict(self, z, X, y=None):
        feed_dict = dict()
        if z is not None:
            feed_dict[self._z] = z
        if X is not None:
            feed_dict[self._X] = X
        if y is not None:
            feed_dict[self._model.y] = y
        return feed_dict

    def _save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self._saver.save(self._sess, os.path.join(checkpoint_dir, self._model_name), global_step=step)
        self._save_obj()

    def _save_obj(self):
        # Saving the objects:
        if not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'], exist_ok=True)

        with open(self.params['save_dir']+'params.pkl', 'wb') as f:
            pickle.dump(self.params, f)
        # with open(self.params['save_dir']+'model.pkl', 'wb') as f:
        #     pickle.dump(self._model, f)
        # with open(self.params['save_dir']+'object.pkl', 'wb') as f:
        #     pickle.dump(self, f, -1)

    def _load(self, checkpoint_dir, file_name = None):
        print(" [*] Reading checkpoints...")
        if file_name:
            self._saver.restore(self._sess, file_name)
            return True

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir, latest_filename=file_name)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            return True

        return False

    def generate(self, N=None, z=None, X=None, sess=None, file_name = None):
        if N and z:
            ValueError('Please choose between N and z')
        if sess is not None:
            self._sess = sess
            return self._generate_sample(N=N, z=z, X=X, file_name=file_name)
        with tf.Session() as self._sess:
            return self._generate_sample(N=N, z=z, X=X, file_name=file_name)

    def _generate_sample(self, N=None , z=None, X=None, file_name=None):
        self._load(self._savedir, file_name=file_name)
        if z is None:
            if N is None:
                N = self.batch_size
        z = self._sample_latent(N)

        y, y_raw = self._generate_sample_safe(z, X)
        return y, y_raw

    def _generate_sample_safe(self, z, X):

        y = []
        y_raw =[]
        N = len(z)
        sind = 0
        bs = self.batch_size
        y_vec = self._get_classes(self.batch_size)
        if N > bs:
            nb = (N-1) // bs
            for i in range(nb):
                if X is not None:
                    y_t, y_raw_t = self._sess.run([self._G_fake, self._G_raw], feed_dict = self._get_dict(z[sind:sind+bs], X[sind:sind+bs], y_vec))
                else:
                    y_t, y_raw_t = self._sess.run([self._G_fake, self._G_raw], feed_dict = self._get_dict(z[sind:sind+bs], None, y_vec))
                y.append(y_t)
                y_raw.append(y_raw_t)
                sind = sind + bs
        if y_vec is None:
            y_sub_vec = None
        else:
            y_sub_vec = y_vec[:N-sind]
        if X is not None:
            y_t, y_raw_t = self._sess.run([self._G_fake, self._G_raw], feed_dict=self._get_dict(z[sind:], X[sind:], y_sub_vec))
        else:
            y_t, y_raw_t = self._sess.run([self._G_fake, self._G_raw], feed_dict=self._get_dict(z[sind:], None, y_sub_vec))
        y.append(y_t)
        y_raw.append(y_raw_t)

        return np.vstack(y), np.vstack(y_raw)

    def _normalize(self, x):
        return (x - self._mean)/self._var

    def _unnormalize(self, x):
        return x * self._var + self._mean

    @property
    def has_encoder(self):
        return self._has_encoder

    @property
    def model_name(self):
        return self._model_name


# class WGAN_upsampler(GAN):

#     def _build_model(self, model):

#         self._G_fake, self.D_real, self.D_fake, Xs, G_fake_s = model(self.params, self._z, self._X)

#         self._G_raw = utils.inv_pre_process(self._G_fake, self.params['k'])
#         self._X_raw = utils.inv_pre_process(self._X, self.params['k'])
        
#         t_vars = tf.trainable_variables()
#         utils.show_all_variables()

#         d_vars = [var for var in t_vars if 'discriminator' in var.name]
#         g_vars = [var for var in t_vars if 'generator' in var.name]

#         gamma_gp = self.params['optimization']['gamma_gp']
#         if not gamma_gp:
#             D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
#             D_gp = tf.constant(0, dtype=tf.float32)
#             print(" [!] Using weight clipping")
#         else:
#             D_clip = tf.constant(0, dtype=tf.float32)
#             # calculate `x_hat`
#             epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0, maxval=1)
#             x_hat = epsilon * self._X + (1.0 - epsilon) * self._G_fake

#             D_x_hat = model.discriminator(x_hat, reuse=True)

#             # gradient penalty
#             gradients = tf.gradients(D_x_hat, [x_hat])

#             D_gp = gamma_gp * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

#         # calculate discriminator's loss
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)

#         e = (Xs -G_fake_s)
#         weight_l2 = self.params['optimization']['weight_l2']
#         reg_l2 = self.params['generator']['latent_dim'] * weight_l2
#         L2_loss = reg_l2 * tf.reduce_mean(e*e)

#         self._D_loss = D_loss_f - D_loss_r + D_gp
#         self._G_loss = -D_loss_f + L2_loss

#         tf.summary.scalar("Disc/Loss", self._D_loss, collections=["Training"])
#         tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
#         tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
#         tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])

#         tf.summary.scalar("Gen/Loss_fake", - D_loss_f, collections=["Training"])
#         tf.summary.scalar("Gen/Loss_l2", L2_loss, collections=["Training"])
#         tf.summary.scalar("Gen/Loss", self._G_loss, collections=["Training"])

#         global_step = tf.Variable(0, name="global_step", trainable=False)


#         optimizer_D, optimizer_G = optimization.build_optmizer(self.params['optimization'])

#         grads_and_vars_d = optimizer_D.compute_gradients(self._D_loss, var_list=d_vars)
#         grads_and_vars_g = optimizer_G.compute_gradients(self._G_loss, var_list=g_vars)

#         self.D_solver = optimizer_D.apply_gradients(grads_and_vars_d, global_step=global_step)
#         self.G_solver = optimizer_G.apply_gradients(grads_and_vars_g, global_step=global_step)

#         optimization.buid_opt_summaries(optimizer_D, 
#                                         grads_and_vars_d, 
#                                         optimizer_G, 
#                                         grads_and_vars_g, 
#                                         self.params['optimization'])


# class WVEEGAN(GAN):

#     def _build_model(self, model):

#         self._G_fake, self.D_real, self.D_fake, z_real, z_fake = model(self.params, self._z, self._X)


#         self._G_raw = utils.inv_pre_process(self._G_fake, self.params['k'])
#         self._X_raw = utils.inv_pre_process(self._X, self.params['k'])
        
#         t_vars = tf.trainable_variables()
#         utils.show_all_variables()

#         d_vars = [var for var in t_vars if 'discriminator' in var.name]
#         g_vars = [var for var in t_vars if 'generator' in var.name]
#         e_vars = [var for var in t_vars if 'encoder' in var.name]

#         gamma_gp = self.params['optimization']['gamma_gp']
#         if not gamma_gp:
#             D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
#             D_gp = tf.constant(0, dtype=tf.float32)
#             print(" [!] Using weight clipping")
#         else:
#             D_clip = tf.constant(0, dtype=tf.float32)
#             # calculate `x_hat`
#             epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0, maxval=1)
#             x_hat = epsilon * self._X + (1.0 - epsilon) * self._G_fake
#             epsilon = tf.reshape(epsilon,shape=[self.batch_size,1])
#             z_hat = epsilon * z_real + (1.0 - epsilon) * self._z
#             D_x_hat = model.discriminator(X=x_hat, z=z_hat, reuse=True)

#             # gradient penalty
#             gradients = tf.gradients(D_x_hat, [x_hat])

#             D_gp = gamma_gp * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

#         # calculate discriminator's loss
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
        
#         e = (self._z - z_fake)
#         weight_l2 = self.params['optimization']['weight_l2']
#         reg_l2 = self.params['generator']['latent_dim'] * weight_l2
#         L2_loss = reg_l2 * tf.reduce_mean(e*e)


#         self._D_loss = D_loss_f - D_loss_r + D_gp
#         self._G_loss = - D_loss_f + L2_loss

#         tf.summary.scalar("Disc/Loss", self._D_loss, collections=["Training"])
#         tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
#         tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
#         tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])

#         tf.summary.scalar("Gen/Loss_fake", - D_loss_f, collections=["Training"])
#         tf.summary.scalar("Gen/Loss_l2", L2_loss, collections=["Training"])
#         tf.summary.scalar("Gen/Loss", self._G_loss, collections=["Training"])


#         global_step = tf.Variable(0, name="global_step", trainable=False)


#         optimizer_D, optimizer_G = optimization.build_optmizer(self.params['optimization'])

#         grads_and_vars_d = optimizer_D.compute_gradients(self._D_loss, var_list=d_vars)
#         grads_and_vars_g = optimizer_G.compute_gradients(self._G_loss, var_list=g_vars+e_vars)

#         self.D_solver = optimizer_D.apply_gradients(grads_and_vars_d, global_step=global_step)
#         self.G_solver = optimizer_G.apply_gradients(grads_and_vars_g, global_step=global_step)

#         optimization.buid_opt_summaries(optimizer_D, 
#                                         grads_and_vars_d, 
#                                         optimizer_G, 
#                                         grads_and_vars_g, 
#                                         self.params['optimization'])


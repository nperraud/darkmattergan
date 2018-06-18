import tensorflow as tf
import numpy as np
from blocks import *


def rprint(msg, reuse=False):
    if not reuse:
        print(msg)


class GanModel(object):
    ''' Abstract class for the model'''
    def __init__(self, params, name='gan', is_3d=False):
        self.name = name
        self.params = params
        self.params['generator']['is_3d'] = is_3d
        self.params['discriminator']['is_3d'] = is_3d    
        self._is_3d = is_3d
        self.G_fake = None
        self.D_real = None
        self.D_fake = None
        self._D_loss = None
        self._G_loss = None

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

    @property
    def is_3d(self):
        return self._is_3d

    @property
    def has_encoder(self):
        return False


class WGanModel(GanModel):
    def __init__(self, params, X, z, name='wgan', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        self.G_fake = self.generator(z, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)


# This is for testing the (expected non positive) effect of normalization on the latent variable
# Use of a regular WGAN if you need a simple Wasserstein GAN
class WNGanModel(GanModel):
    def __init__(self, params, X, z, name='wngan', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        zn = tf.nn.l2_normalize(z, 1)
        self.G_fake = self.generator(zn, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        self._D_loss = D_loss_r - D_loss_f + D_gp
        self._G_loss = D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)


class CondWGanModel(GanModel):
    def __init__(self, params, X, z, name='CondWGan', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.G_fake = self.generator(z, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], y=self.y, reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], z=self.y, reuse=reuse)


class TemporalGanModel(GanModel):
    def __init__(self, params, X, z, name='TempWGanV1', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        zn = tf.nn.l2_normalize(z, 1)
        z_shape = tf.shape(zn)
        scaling = (np.arange(params['num_classes']) + 1) / params['num_classes']
        scaling = np.resize(scaling, (params['optimization']['batch_size'], 1))
        y = tf.constant(scaling, dtype=tf.float32, name='y')
        y = y[:z_shape[0]]
        zn = tf.multiply(zn, y)

        self.G_fake = self.generator(zn, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], z=self.y, reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


class TempGanModelv2(GanModel):
    def __init__(self, params, X, z, name='TempWGanV2', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        zn = tf.nn.l2_normalize(z, 1)
        z_shape = tf.shape(zn)
        scaling = (np.arange(params['num_classes']) + 1) / params['num_classes']
        scaling = np.resize(scaling, (params['optimization']['batch_size'], 1))
        y = tf.constant(scaling, dtype=tf.float32, name='y')
        y = y[:z_shape[0]]
        zn = tf.multiply(zn, y)

        self.G_fake = self.generator(zn, reuse=False)

        self.D_real = self.discriminator(X, reuse=False)
        self.D_c_real = self.c_discriminator(X, reuse=False)

        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        self.D_c_fake = self.c_discriminator(self.G_fake, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        D_c_loss_f = tf.reduce_mean(self.D_c_fake)
        D_c_loss_r = tf.reduce_mean(self.D_c_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        D_c_gp = wgan_regularization(gamma_gp, self.c_discriminator, [self.G_fake], [X])

        self._D_loss = D_loss_f - D_loss_r + D_c_loss_f - D_c_loss_r + D_gp + D_c_gp
        self._G_loss = -D_loss_f -D_c_loss_f

        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], z=self.y, reuse=reuse)

    def c_discriminator(self, X, reuse):
        bs = self.params['optimization']['batch_size']
        nc = self.params['num_classes']
        seq = np.repeat(np.arange(bs // nc) * nc, nc - 1) + np.tile(np.arange(nc - 1), bs // nc)
        x1 = tf.gather(X, seq)
        x2 = tf.gather(X, seq + 1)
        x = tf.concat([x1,x2], axis=3)
        return discriminator(x, self.params['discriminator'], reuse=reuse, scope="consistency_discriminator")

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


class TemporalGanModelv3(GanModel):
    def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        assert 'time' in params.keys()

        zn = tf.nn.l2_normalize(z, 1) * np.sqrt(params['generator']['latent_dim'])
        z_shape = tf.shape(zn)
        scaling = np.asarray(params['time']['class_weights'])
        gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
        scaling = np.resize(scaling, (gen_bs, 1))
        default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
        self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
        t = self.y[:z_shape[0]]
        zn = tf.multiply(zn, t)

        self.G_c_fake = self.generator(zn, reuse=False)
        self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

        if params['time']['use_diff_stats']:
            self.disc = self.df_discriminator
        else:
            self.disc = self.discriminator

        self.D_real = self.disc(X, reuse=False)
        self.D_fake = self.disc(self.G_fake, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.disc, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def reshape_time_to_channels_old(self, X):
        bs = self.params['optimization']['batch_size']
        nc = self.params['time']['num_classes']
        idx = np.arange(bs) * nc
        x = tf.gather(X, idx)
        #for i in (np.arange(nc - 1) + 1):
        for i in range(1, nc):
            x = tf.concat([x, tf.gather(X, idx + i)], axis=3)
        return x

    def reshape_time_to_channels(self, X):
        nc = self.params['time']['num_classes']
        lst = []
        for i in range(nc):
            lst.append(X[i::nc])
        return tf.concat(lst, axis=3)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)

    def df_discriminator(self, X, reuse):
        y = X[:, :, :, 1:] - X[:, :, :, :-1]
        return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


class TemporalGanModelv4(GanModel):
    def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        assert 'time' in params.keys()

        z0 = z[:, 0::2]
        z1 = z[:, 1::2]
        zn = tf.nn.l2_normalize(z1, 1) * np.sqrt(params['generator']['latent_dim'] / 2)
        z_shape = tf.shape(zn)
        scaling = np.asarray(params['time']['class_weights'])
        gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
        scaling = np.resize(scaling, (gen_bs, 1))
        default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
        self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
        t = self.y[:z_shape[0]]
        zn = tf.multiply(zn, t)
        zn = tf.expand_dims(zn, -1)
        z0 = tf.expand_dims(z0, -1)
        zn = tf.concat([z0, zn], axis=2)

        self.G_c_fake = self.generator(zn, reuse=False)
        self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

        if params['time']['use_diff_stats']:
            self.disc = self.df_discriminator
        else:
            self.disc = self.discriminator

        self.D_real = self.disc(X, reuse=False)
        self.D_fake = self.disc(self.G_fake, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.disc, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def reshape_time_to_channels_old(self, X):
        bs = self.params['optimization']['batch_size']
        nc = self.params['time']['num_classes']
        idx = np.arange(bs) * nc
        x = tf.gather(X, idx)
        #for i in (np.arange(nc - 1) + 1):
        for i in range(1, nc):
            x = tf.concat([x, tf.gather(X, idx + i)], axis=3)
        return x

    def reshape_time_to_channels(self, X):
        nc = self.params['time']['num_classes']
        lst = []
        for i in range(nc):
            lst.append(X[i::nc])
        return tf.concat(lst, axis=3)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)

    def df_discriminator(self, X, reuse):
        y = X[:, :, :, 1:] - X[:, :, :, :-1]
        return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


class TemporalGanModelv5(GanModel):
    def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        assert 'time' in params.keys()

        z_shape = tf.shape(z)
        scaling = np.asarray(params['time']['class_weights'])
        gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
        scaling = np.resize(scaling, (gen_bs, 1))
        default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
        self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
        t = self.y[:z_shape[0]]

        self.G_c_fake = self.generator(zn, reuse=False)
        self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

        if params['time']['use_diff_stats']:
            self.disc = self.df_discriminator
        else:
            self.disc = self.discriminator

        self.D_real = self.disc(X, reuse=False)
        self.D_fake = self.disc(self.G_fake, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.disc, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def reshape_time_to_channels_old(self, X):
        bs = self.params['optimization']['batch_size']
        nc = self.params['time']['num_classes']
        idx = np.arange(bs) * nc
        x = tf.gather(X, idx)
        #for i in (np.arange(nc - 1) + 1):
        for i in range(1, nc):
            x = tf.concat([x, tf.gather(X, idx + i)], axis=3)
        return x

    def reshape_time_to_channels(self, X):
        nc = self.params['time']['num_classes']
        lst = []
        for i in range(nc):
            lst.append(X[i::nc])
        return tf.concat(lst, axis=3)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)

    def df_discriminator(self, X, reuse):
        y = X[:, :, :, 1:] - X[:, :, :, :-1]
        return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

class TemporalGanModelv3E(GanModel):
    def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        assert 'time' in params.keys()

        zn = tf.nn.l2_normalize(z, 1) * np.sqrt(params['generator']['latent_dim'])
        z_shape = tf.shape(zn)
        scaling = np.asarray(params['time']['class_weights'])
        gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
        scaling = np.resize(scaling, (gen_bs, 1))
        default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
        self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
        t = self.y[:z_shape[0]]
        zn = tf.multiply(zn, t)

        self.G_c_fake = self.generator(zn, reuse=False)
        self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)

        enc = self.encoder(self.G_c_fake, reuse=False)
        # Perhaps something should be added to the loss,
        # as we know how it should be encoding time?
        self._E_loss = tf.reduce_mean(tf.square(enc - zn))

        # This should make it easy to compare images with their encoded and decoded counterparts
        width = self.params['image_size'][0]
        self.single_images = tf.placeholder(tf.float32, shape=[None, width, width, 1])
        self.encoded_images = self.encoder(self.single_images, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])

        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def reshape_time_to_channels(self, X):
        bs = self.params['optimization']['batch_size']
        nc = self.params['time']['num_classes']
        idx = np.arange(bs) * nc
        x = tf.gather(X, idx)
        #for i in (np.arange(nc - 1) + 1):
        for i in range(1, nc):
            x = tf.concat([x, tf.gather(X, idx + i)], axis=3)
        return x

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)

    def encoder(self, X, reuse):
        return encoder(X, self.params['encoder'], self.latent_dim, reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

    @property
    def E_loss(self):
        return self._E_loss

    @property
    def has_encoder(self):
        return True


class TemporalGanModelv4old(GanModel):
    def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        assert 'time' in params.keys()

        zn = tf.nn.l2_normalize(z, 1) * np.sqrt(params['generator']['latent_dim'])
        z_shape = tf.shape(zn)
        scaling = np.asarray(params['time']['class_weights'])
        gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
        scaling = np.resize(scaling, (gen_bs, 1))
        default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
        self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
        t = self.y[:z_shape[0]]
        zn = tf.multiply(zn, t)

        self.G_c_fake = self.generator(zn, reuse=False)
        self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)

        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)

        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def reshape_time_to_channels(self, X):
        bs = self.params['optimization']['batch_size']
        nc = self.params['time']['num_classes']
        idx = np.arange(bs) * nc
        x = tf.gather(X, idx)
        #for i in (np.arange(nc - 1) + 1):
        for i in range(1, nc):
            x = tf.concat([x, tf.gather(X, idx + i)], axis=3)
        return x

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse, dif_stats=False):
        y = X[:,:,:,1:] - X[:,:,:,:-1]
        return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


def reshape_channels_to_separate(self, X):
    t = tf.transpose(X, [0, 3, 1, 2])
    shape = tf.shape(t)
    return tf.reshape(t, [shape[0]*shape[1], shape[2], shape[3], 1])


class WVeeGanModel(GanModel):
    def __init__(self, params, X, z, name='veegan', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        self.latent_dim = params['generator']['latent_dim']
        self.G_fake = self.generator(z, reuse=False)
        self.z_real = self.encoder(X=X, reuse=False)
        self.D_real = self.discriminator(X=X, z=self.z_real, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, z=z, reuse=True)
        self.z_fake = self.encoder(X=self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.z_fake], [X, self.z_real])

        e = (z - self.z_fake)
        weight_l2 = self.params['optimization']['weight_l2']
        reg_l2 = self.latent_dim * weight_l2
        L2_loss = reg_l2 * tf.reduce_mean(tf.square(e))

        self._D_loss = D_loss_f - D_loss_r + D_gp
        self._G_loss = -D_loss_f + L2_loss
        self._E_loss = L2_loss

        tf.summary.scalar("Enc/Loss_l2", self._E_loss, collections=["Training"])
        tf.summary.scalar("Gen/Loss_f", -D_loss_f, collections=["Training"])

        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, -D_loss_r)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, z, reuse):
        return discriminator(X, self.params['discriminator'], z=z, reuse=reuse)

    def encoder(self, X, reuse):
        return encoder(X, self.params['encoder'], self.latent_dim, reuse=reuse)

    @property
    def E_loss(self):
        return self._E_loss

    @property
    def has_encoder(self):
        return True


class LapGanModel(GanModel):
    def __init__(self, params, X, z, name='lapgan', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        self.upsampling = params['generator']['upsampling']
        self.Xs = down_sampler(X, s=self.upsampling)
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')       
        self.Xsu = up_sampler(self.Xs, s=self.upsampling)
        self.G_fake = self.generator(X=self.Xsu, z=z, reuse=False)
        # self.D_real = self.discriminator(X-self.Xsu, self.Xsu, reuse=False)
        # self.D_fake = self.discriminator(self.G_fake-self.Xsu, self.Xsu, reuse=True)
        self.D_real = self.discriminator(X, self.Xsu, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])

    def generator(self, X, z, reuse):
        return generator_up(X, z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


class LapGanModelTanh(GanModel):
    def __init__(self, params, X, z, name='lapgan', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        self.upsampling = params['generator']['upsampling']
        self.Xs = down_sampler(X, s=self.upsampling)
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')       
        self.Xsu = up_sampler(self.Xs, s=self.upsampling)
        G_tmp = self.generator(X=self.y, z=z, reuse=False)
        self.G_fake = tf.tanh(G_tmp)
        # self.D_real = self.discriminator(X-self.Xsu, self.Xsu, reuse=False)
        # self.D_fake = self.discriminator(self.G_fake-self.Xsu, self.Xsu, reuse=True)
        self.D_real = -self.discriminator(tf.atanh(X), tf.atanh(self.Xsu), reuse=False)
        self.D_fake = self.discriminator(G_tmp, tf.atanh(self.Xsu), reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])

    def generator(self, X, z, reuse):
        return generator_up(X, z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


class Gan12Model(GanModel):
    def __init__(self, params, X, z, name='wgan12', is_3d=False):
        super().__init__(params=params, name=name, is_3d=is_3d)
        X1, _ = tf.split(X, 2, axis = params['generator']['border']['axis'])
        self.G_fake = self.generator(z, X1, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def generator(self, z, X, reuse):
        return generator12(z, X, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)


class LapPatchWGANModel(GanModel):
    """4 different generators, probably not a good idea. Need too much training time. Not so good results."""
    def __init__(self, params, X, z, name='lapgan', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        
        # A) Down sampling the image
        self.upsampling = params['generator']['upsampling']
        self.Xs = down_sampler(X, s=self.upsampling)

        # The input is the downsampled image
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')

        # B) Split the image in 4 parts
        top, bottom = tf.split(self.y, 2, axis=1)
        self.Xs1, self.Xs2 = tf.split(top, 2, axis=2)
        self.Xs3, self.Xs4 = tf.split(bottom, 2, axis=2)

        # B') Split latent in 4 parts
        # This may/should be done differently?
        bs = tf.shape(self.y)[0]  # Batch size
        z = tf.reshape(z, [bs, *inshape])
        topz, bottomz = tf.split(z, 2, axis=1)
        z1, z2 = tf.split(topz, 2, axis=2)
        z3, z4 = tf.split(bottomz, 2, axis=2)

        # C) Define the 4 Generators

        self.G_fake1 = self.generator(X=self.Xs1, z=z1, reuse=False, scope='generator1')
        y1 = tf.reverse(self.G_fake1, axis=[2])
        self.G_fake2 = self.generator(X=self.Xs2, z=z2, y=y1, reuse=False, scope='generator2')
        y21 = tf.reverse(self.G_fake1, axis=[1])
        y22 = tf.reverse(self.G_fake2, axis=[1,2])
        y2 = tf.concat([y21, y22], axis=3)
        self.G_fake3 = self.generator(X=self.Xs3, z=z3,y=y2, reuse=False, scope='generator3')
        y31 = tf.reverse(self.G_fake1, axis=[1,2])
        y32 = tf.reverse(self.G_fake2, axis=[1])
        y33 = tf.reverse(self.G_fake3, axis=[2])
        y3 = tf.concat([y31, y32, y33], axis=3)
        self.G_fake4 = self.generator(X=self.Xs4, z=z4, y=y3, reuse=False, scope='generator4')

        # D) Concatenate back
        top = tf.concat([self.G_fake1,self.G_fake2], axis=2)
        bottom = tf.concat([self.G_fake3,self.G_fake4], axis=2)
        self.G_fake = tf.concat([top,bottom], axis=1)

        # E) Discriminator
        self.Xsu = up_sampler(self.y, s=self.upsampling)
        self.D_real = self.discriminator(X, self.Xsu, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
        if True:
            tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y1", y1, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

    def generator(self, X, z, reuse, scope, y=None):
        return generator_up(X, z, self.params['generator'], y=y, reuse=reuse, scope=scope)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


class LapPatchWGANsingleModel(GanModel):
    """Seems to work fine but is recursive, so might be a bit slow."""
    def __init__(self, params, X, z, name='lappatchsingle', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        
        # A) Down sampling the image
        self.upsampling = params['generator']['upsampling']
        self.Xs = down_sampler(X, s=self.upsampling)

        # The input is the downsampled image
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
        self.Xsu = up_sampler(self.y, s=self.upsampling)

        # B) Split the image in 4 parts
        top, bottom = tf.split(self.Xsu, 2, axis=1)
        self.Xs1, self.Xs2 = tf.split(top, 2, axis=2)
        self.Xs3, self.Xs4 = tf.split(bottom, 2, axis=2)

        # B') Split latent in 4 parts
        # This may/should be done differently?
        bs = tf.shape(self.y)[0]  # Batch size
        zshape = X.shape.as_list()[1:]
        z = tf.reshape(z, [bs, *zshape])
        topz, bottomz = tf.split(z, 2, axis=1)
        z1, z2 = tf.split(topz, 2, axis=2)
        z3, z4 = tf.split(bottomz, 2, axis=2)

        # C) Define the Generator

        tinshape = tf.shape(z1)
        y00 = tf.fill(tinshape, -1.)
        y0 = tf.concat([y00, y00, y00], axis=3)

        self.G_fake1 = self.generator(X=self.Xs1, z=z1, border=y0, reuse=False, scope='generator')
        y11 = tf.reverse(self.G_fake1, axis=[2])
        y1 = tf.concat([y11, y00, y00], axis=3)

        self.G_fake2 = self.generator(X=self.Xs2, z=z2, border=y1, reuse=True, scope='generator')
        y21 = tf.reverse(self.G_fake1, axis=[1])
        y22 = tf.reverse(self.G_fake2, axis=[1, 2])
        y2 = tf.concat([y21, y22, y00], axis=3)

        self.G_fake3 = self.generator(X=self.Xs3, z=z3, border=y2, reuse=True, scope='generator')
        y31 = tf.reverse(self.G_fake1, axis=[1, 2])
        y32 = tf.reverse(self.G_fake2, axis=[1])
        y33 = tf.reverse(self.G_fake3, axis=[2])
        y3 = tf.concat([y31, y32, y33], axis=3)
        self.G_fake4 = self.generator(X=self.Xs4, z=z4, border=y3, reuse=True, scope='generator')

        # D) Concatenate back
        top = tf.concat([self.G_fake1, self.G_fake2], axis=2)
        bottom = tf.concat([self.G_fake3, self.G_fake4], axis=2)
        self.G_fake = tf.concat([top, bottom], axis=1)

        # E) Discriminator
        self.D_real = self.discriminator(X, self.Xsu, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
        if True:
            tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y1", y11, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

    def generator(self, X, z, border, reuse, scope):
        return generator_up(tf.concat([X, border], axis=3), z, self.params['generator'], reuse=reuse, scope=scope)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


class PatchWGANsingleModel(GanModel):
    '''
    Divide image into 4 parts, and iterative generate them
    '''
    def __init__(self, params, X, z, name='patchsingle', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)

        # A) Split latent in 4 parts
        bs = tf.shape(X)[0]  # Batch size
        # nb pixel
        inshape = X.shape.as_list()[1:]
        z = tf.reshape(z, [bs, *inshape])
        topz, bottomz = tf.split(z, 2, axis=1)
        z1, z2 = tf.split(topz, 2, axis=2)
        z3, z4 = tf.split(bottomz, 2, axis=2)

        tinshape = tf.shape(z1)

        y00 = tf.fill(tinshape, -1.)
        y0 = tf.concat([y00, y00, y00], axis=3)
        self.G_fake1 = self.generator(z=z1, border=y0, reuse=False, scope='generator')

        y11 = tf.reverse(self.G_fake1, axis=[2])
        y1 = tf.concat([y11, y00, y00], axis=3)
        self.G_fake2 = self.generator(z=z2, border=y1, reuse=True, scope='generator')

        y21 = tf.reverse(self.G_fake1, axis=[1])
        y22 = tf.reverse(self.G_fake2, axis=[1,2])
        y2 = tf.concat([y21, y22, y00], axis=3)
        self.G_fake3 = self.generator(z=z3,border=y2, reuse=True, scope='generator')

        y31 = tf.reverse(self.G_fake1, axis=[1,2])
        y32 = tf.reverse(self.G_fake2, axis=[1])
        y33 = tf.reverse(self.G_fake3, axis=[2])
        y3 = tf.concat([y31, y32, y33], axis=3)
        self.G_fake4 = self.generator(z=z4, border=y3, reuse=True, scope='generator')

        # B) Concatenate back
        top = tf.concat([self.G_fake1,self.G_fake2], axis=2)
        bottom = tf.concat([self.G_fake3,self.G_fake4], axis=2)
        self.G_fake = tf.concat([top,bottom], axis=1)

        # C) Discriminator
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)

        # D) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # E) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Real", X, max_outputs=2, collections=['Images'])
        tf.summary.image("training/G_fake", self.G_fake, max_outputs=2, collections=['Images'])
        if True:
            tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y1", y11, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
            tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

    def generator(self, z, border, reuse, scope):
        '''
        X = down-sampled information
        y = border information
        '''
        return generator_up(X=border, z=z, params=self.params['generator'], reuse=reuse, scope=scope)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)


class LapPatchWGANsimpleModel(GanModel):
    def __init__(self, params, X, z, name='lapgansimple', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        
        # A) Down sampling the image
        self.upsampling = params['generator']['upsampling']

        X0, border = tf.split(X, [1,3],axis=3)
        self.Xs = down_sampler(X0, s=self.upsampling)

        # The input is the downsampled image
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
        # The border is a different input
        inshape = border.shape.as_list()[1:]
        self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
        X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
        X1f = tf.reverse(X1, axis=[1])
        X2f = tf.reverse(X2, axis=[2])
        X3f = tf.reverse(X3, axis=[1,2])
        flip_border = tf.concat([X1f, X2f, X3f], axis=3)
        self.Xsu = up_sampler(self.y, s=self.upsampling)

        self.G_fake = self.generator(X=self.Xsu, z=z, border=flip_border, reuse=False, scope='generator')


        # E) Discriminator
        self.D_real = self.discriminator(X0, self.Xsu, flip_border, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, self.Xsu, flip_border, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu, flip_border], [X0, self.Xsu, flip_border])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X0 - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
        if True:
            # D) Concatenate back
            top = tf.concat([X3,X2], axis=1)
            bottom = tf.concat([X1,X0], axis=1)
            bottom_g = tf.concat([X1,self.G_fake], axis=1)
            full_img = tf.concat([top,bottom], axis=2)
            full_img_fake = tf.concat([top,bottom_g], axis=2)
            tf.summary.image("training/full_img_real", full_img, max_outputs=4, collections=['Images'])
            tf.summary.image("training/full_img_fake", full_img_fake, max_outputs=4, collections=['Images'])
            tf.summary.image("training/X0", X0, max_outputs=2, collections=['Images'])
            tf.summary.image("training/X1", X1, max_outputs=1, collections=['Images'])
            tf.summary.image("training/X2", X2, max_outputs=1, collections=['Images'])
            tf.summary.image("training/X3", X3, max_outputs=1, collections=['Images'])
            tf.summary.image("training/X1f", X1f, max_outputs=1, collections=['Images'])
            tf.summary.image("training/X2f", X2f, max_outputs=1, collections=['Images'])
            tf.summary.image("training/X3f", X3f, max_outputs=1, collections=['Images'])

    def generator(self, X, z, border, reuse, scope):
        return generator_up(tf.concat([X, border], axis=3), z, params=self.params['generator'], y=None, reuse=reuse, scope=scope)

    def discriminator(self, X, Xsu, border, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu, border], axis=3), self.params['discriminator'], reuse=reuse)


class LapPatchWGANsimpleUnfoldModel(GanModel):
    def __init__(self, params, X, z, name='lapgansimpleunfold', is_3d=False):
        ''' z must have the same dimension as X'''
        super().__init__(params=params, name=name, is_3d=is_3d)
        
        # A) Down sampling the image
        self.upsampling = params['generator']['upsampling']

        X0, border = tf.split(X, [1,3], axis=3)
        self.Xs = down_sampler(X0, s=self.upsampling)

        # The input is the downsampled image
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
       
        # The border is a different input
        inshape = border.shape.as_list()[1:]
        self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
        X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
        X1f = tf.reverse(X1, axis=[1])
        X2f = tf.reverse(X2, axis=[2])
        X3f = tf.reverse(X3, axis=[1,2])
        flip_border = tf.concat([X1f, X2f, X3f], axis=3)

        self.G_fake = self.generator(y=up_sampler(self.y, s=self.upsampling),
                                     z=z,
                                     border=flip_border,
                                     reuse=False,
                                     scope='generator')

        # D) Concatenate back
        top = tf.concat([X3,X2], axis=1)
        bottom = tf.concat([X1,X0], axis=1)
        bottom_g = tf.concat([X1,self.G_fake], axis=1)
        X_real = tf.concat([top,bottom], axis=2)
        G_fake = tf.concat([top,bottom_g], axis=2)
        Xs_full = down_sampler(X_real, s=self.upsampling)
        self.Xsu = up_sampler(Xs_full, s=self.upsampling)

        # E) Discriminator
        self.D_real = self.discriminator(X_real, self.Xsu, reuse=False)
        self.D_fake = self.discriminator(G_fake, self.Xsu, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake, self.Xsu], [X_real, self.Xsu])
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Real_full_image", X_real, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_full_image", G_fake, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Downsample_X0", self.y, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Real_Diff", X_real - self.Xsu, max_outputs=1, collections=['Images'])
        tf.summary.image("training/Fake_Diff", G_fake - self.Xsu, max_outputs=1, collections=['Images'])
        if True:
            tf.summary.image("checking/X0", X0, max_outputs=2, collections=['Images'])
            tf.summary.image("checking/X1", X1, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X2", X2, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X3", X3, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X1f", X1f, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X2f", X2f, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X3f", X3f, max_outputs=1, collections=['Images'])

    def generator(self, y, z, border, reuse, scope):
        return generator_up(tf.concat([y, border], axis=3), z, self.params['generator'], y=None, reuse=reuse, scope=scope)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


class upscale_WGAN_pixel_CNN(GanModel):
    '''
    Generate blocks, using top, left and top-left border information
    '''
    def __init__(self, params, X, z, name='upscale_WGAN_pixel_CNN', is_3d=False):
        ''' z must have the same dimension as X, or downsampled X in case of downsampling'''
        super().__init__(params=params, name=name, is_3d=is_3d)

        # A) Get downsampling factor
        self.downsampling = params['generator'].get('downsampling', None)

        if self.is_3d:
            self.__init_3d(params, X, z)

        else:
            self.__init_2d(params, X, z)

    def __init_3d(self, params, X, z):
        '''
        build the computation graph for 3d case.
        '''

        # A) Separate real data and border information
        real, border = tf.split(X, [1,7], axis=4)

        if self.downsampling:
            self.real_downsampled = down_sampler(real, s=self.downsampling, is_3d=True)
            inshape = self.real_downsampled.shape.as_list()[1:]
            # The input is the downsampled image
            self.downsampled = tf.placeholder_with_default(self.real_downsampled, shape=[None, *inshape], name='downsampled')
        else:
            self.downsampled = None

        # B) Split the borders
        inshape = border.shape.as_list()[1:]
        self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
        d_above, d_left, d_corner, up, u_above, u_left, u_corner = tf.split(self.border, [1, 1, 1, 1, 1, 1, 1], axis=4)

        # C) Flip the borders for proper alignment with original data
        flip_d_above  = tf.reverse( d_above, axis=[2])
        flip_d_left   = tf.reverse(  d_left, axis=[3])
        flip_d_corner = tf.reverse(d_corner, axis=[2, 3])
        flip_up       = tf.reverse(      up, axis=[1])
        flip_u_above  = tf.reverse( u_above, axis=[1, 2])
        flip_u_left   = tf.reverse(  u_left, axis=[1, 3])
        flip_u_corner = tf.reverse(u_corner, axis=[1, 2, 3])

        flip_border = tf.concat([flip_d_above, flip_d_left, flip_d_corner, flip_up, flip_u_above, flip_u_left, flip_u_corner], axis=4)

        self.G_fake = self.generator(downsampled=self.downsampled, z=z, border=flip_border, reuse=False, scope='generator')

        # D) Concatenate back
        down_left     = tf.concat([ d_corner,      d_left], axis=2)
        down_right    = tf.concat([  d_above,        real], axis=2)
        down_right_g  = tf.concat([  d_above, self.G_fake], axis=2)
        
        down_cuboid     = tf.concat([down_left,   down_right], axis=3)
        down_cuboid_g   = tf.concat([down_left, down_right_g], axis=3)

        up_left   = tf.concat([u_corner,   u_left], axis=2)
        up_right  = tf.concat([ u_above,       up], axis=2)
        up_cuboid = tf.concat([ up_left, up_right], axis=3)

        X_real = tf.concat([up_cuboid,   down_cuboid], axis=1)
        G_fake = tf.concat([up_cuboid, down_cuboid_g], axis=1)
        
        if self.downsampling:
            X_down = down_sampler(X_real, s=self.downsampling, is_3d=True)
            X_down_up = up_sampler(X_down, s=self.downsampling, is_3d=True)
        else:
            X_down_up = None

        # E) Discriminator
        self.D_real = self.discriminator(X_real, X_down_up, reuse=False)
        self.D_fake = self.discriminator(G_fake, X_down_up, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']

        if self.downsampling:
            D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake, X_down_up], [X_real, X_down_up])
        else:
            D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake], [X_real])

        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

    def __init_2d(self, params, X, z):
        '''
        build the computation graph for 2d case.
        '''

        # A) Separate real data and border information
        real, border = tf.split(X, [1,3],axis=3)

        if self.downsampling:
            self.real_downsampled = down_sampler(real, s=self.downsampling, is_3d=False)
            inshape = self.real_downsampled.shape.as_list()[1:]
            # The input is the downsampled image
            self.downsampled = tf.placeholder_with_default(self.real_downsampled, shape=[None, *inshape], name='downsampled')
        else:
            self.downsampled = None

        # B) Split the borders
        inshape = border.shape.as_list()[1:]
        self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
        above, left, corner = tf.split(self.border, [1,1,1],axis=3)

        # C) Flip the borders for proper alignment with original data
        flip_above  = tf.reverse( above, axis=[1])
        flip_left   = tf.reverse(  left, axis=[2])
        flip_corner = tf.reverse(corner, axis=[1,2])

        flip_border = tf.concat([flip_above, flip_left, flip_corner], axis=3)

        self.G_fake = self.generator(downsampled=self.downsampled, z=z, border=flip_border, reuse=False, scope='generator')

        # D) Concatenate back
        left    = tf.concat([corner,        left], axis=1)
        right   = tf.concat([ above,        real], axis=1)
        right_g = tf.concat([ above, self.G_fake], axis=1)
        
        X_real = tf.concat([left,  right], axis=2)
        G_fake = tf.concat([left,right_g], axis=2)

        if self.downsampling:
            X_down = down_sampler(X_real, s=self.downsampling, is_3d=False)
            X_down_up = up_sampler(X_down, s=self.downsampling, is_3d=False)
        else:
            X_down_up = None

        # E) Discriminator
        self.D_real = self.discriminator(X_real, X_down_up, reuse=False)
        self.D_fake = self.discriminator(G_fake, X_down_up, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']

        if self.downsampling:
            D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake, X_down_up], [X_real, X_down_up])
        else:
            D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake], [X_real])

        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Real_full_image", X_real, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_full_image", G_fake, max_outputs=2, collections=['Images'])

        if True:
            tf.summary.image("checking/real",               real, max_outputs=2, collections=['Images'])
            tf.summary.image("checking/above",             above, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/left",               left, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/corner",           corner, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/flip_above",   flip_above, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/flip_left",     flip_left, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/flip_corner", flip_corner, max_outputs=1, collections=['Images'])

    def generator(self, downsampled, z, border, reuse, scope):
        return generator_up(downsampled, z, self.params['generator'], y=border, reuse=reuse, scope=scope)

    def discriminator(self, X, X_down_up=None, reuse=True):
        if self.is_3d:
            concat_axis = 4
        else:
            concat_axis = 3

        if self.downsampling:
            return discriminator(tf.concat([X, X_down_up, X-X_down_up], axis=concat_axis), self.params['discriminator'], reuse=reuse)
        else:
            return discriminator(X, self.params['discriminator'], reuse=reuse)


class LapPatchWGANDirect(GanModel):
    def __init__(self, params, X, z, name='lapgandirect', is_3d=False):
        '''Some model for Ankit to try.
        
        z must have the same dimension as X.
        stride of 1
        '''
        super().__init__(params=params, name=name, is_3d=is_3d)
        
        # A) Down sampling the image
        self.upsampling = params['generator']['upsampling']

        X0, border = tf.split(X, [1,3],axis=3)

        # The border is a different input
        inshape = border.shape.as_list()[1:]
        self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
        X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
        X1f = tf.reverse(X1, axis=[1])
        X2f = tf.reverse(X2, axis=[2])
        X3f = tf.reverse(X3, axis=[1,2])
        flip_border = tf.concat([X1f, X2f, X3f], axis=3)
        self.G_fake = self.generator(y=flip_border, z=z, reuse=False, scope='generator')

        # D) Concatenate back
        top = tf.concat([X3,X2], axis=1)
        bottom = tf.concat([X1,X0], axis=1)
        bottom_g = tf.concat([X1,self.G_fake], axis=1)
        X_real = tf.concat([top,bottom], axis=2)
        G_fake = tf.concat([top,bottom_g], axis=2)

        # E) Discriminator
        self.D_real = self.discriminator(X_real, reuse=False)
        self.D_fake = self.discriminator(G_fake, reuse=True)

        # F) Losses
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake], [X_real])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
        # Min(D_loss_r - D_loss_f) = Min -D_loss_f
        self._D_loss = -(D_loss_r - D_loss_f) + D_gp
        self._G_loss = -D_loss_f

        # G) Summaries
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
        tf.summary.image("training/Real_full_image", X_real, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Fake_full_image", G_fake, max_outputs=2, collections=['Images'])
        tf.summary.image("training/Downsample_X0", self.y, max_outputs=2, collections=['Images'])
        if True:
            tf.summary.image("checking/X0", X0, max_outputs=2, collections=['Images'])
            tf.summary.image("checking/X1", X1, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X2", X2, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X3", X3, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X1f", X1f, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X2f", X2f, max_outputs=1, collections=['Images'])
            tf.summary.image("checking/X3f", X3f, max_outputs=1, collections=['Images'])

    def generator(self, y, z, reuse, scope):
        return generator_up(X, z, self.params['generator'], reuse=reuse, scope=scope)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)


# class GanUpSampler(object):
#     def __init__(self, name='gan_upsampler'):
#         self.name = name
#     def generator(self, X, z, reuse):
#         return generator_up(X, z, self.params['generator'], reuse=reuse)
#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)
#     def __call__(self, params, z, X):
#         self.params = params
#         self.upsampling = params['generator']['upsampling']
#         Xs = down_sampler(X, s=self.upsampling)
#         G_fake = self.generator(X=Xs, z=z, reuse=False)
#         G_fake_s = down_sampler(G_fake, s=self.upsampling)
#         D_real = self.discriminator(X, reuse=False)
#         D_fake = self.discriminator(G_fake, reuse=True)       
        
#         return G_fake, D_real, D_fake, Xs, G_fake_s

def wgan_summaries(D_loss, G_loss, D_loss_f, D_loss_r):
    tf.summary.scalar("Disc/Neg_Loss", -D_loss, collections=["Training"])
    tf.summary.scalar("Disc/Neg_Critic", D_loss_f - D_loss_r, collections=["Training"])
    tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
    tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
    tf.summary.scalar("Gen/Loss", G_loss, collections=["Training"])


def fisher_gan_regularization(D_real, D_fake, rho=1):
    with tf.variable_scope("discriminator", reuse=False):
        phi = tf.get_variable('lambda', shape=[],
            initializer=tf.initializers.constant(value=1.0, dtype=tf.float32))
        D_real2 = tf.reduce_mean(tf.square(D_real))
        D_fake2 = tf.reduce_mean(tf.square(D_fake))
        constraint = 1.0 - 0.5 * (D_real2 + D_fake2)

        # Here phi should be updated using another opotimization scheme
        reg_term = phi * constraint + 0.5 * rho * tf.square(constraint)
        print(D_real.shape)
        print(D_real2.shape)        
        print(constraint.shape)
        print(reg_term.shape)
    tf.summary.scalar("Disc/constraint", reg_term, collections=["Training"])
    tf.summary.scalar("Disc/reg_term", reg_term, collections=["Training"])
    return reg_term


def wgan_regularization(gamma, discriminator, list_fake, list_real):
    if not gamma:
        # I am not sure this part or the code is still useful
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        D_gp = tf.constant(0, dtype=tf.float32)
        print(" [!] Using weight clipping")
    else:
        D_clip = tf.constant(0, dtype=tf.float32)
        # calculate `x_hat`
        assert(len(list_fake) == len(list_real))
        bs = tf.shape(list_fake[0])[0]
        eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)

        x_hat = []
        for fake, real in zip(list_fake, list_real):
            singledim = [1]* (len(fake.shape.as_list())-1)
            eps = tf.reshape(eps, shape=[bs,*singledim])
            x_hat.append(eps * real + (1.0 - eps) * fake)

        D_x_hat = discriminator(*x_hat, reuse=True)

        # gradient penalty
        gradients = tf.gradients(D_x_hat, x_hat)

        D_gp = gamma * tf.square(tf.norm(gradients[0], ord=2) - 1.0)
        tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])
    return D_gp


def get_conv(is_3d=False):
    if is_3d:
        conv = conv3d
    else:
        conv = conv2d
    return conv


def deconv(in_tensor, bs, sx, n_filters, shape, stride, summary, conv_num, is_3d=False):
    if is_3d:
        output_shape = [bs, sx, sx, sx, n_filters]
        out_tensor = deconv3d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_3d'.format(conv_num),
                              summary=summary)
    else:
        output_shape = [bs, sx, sx, n_filters]
        out_tensor = deconv2d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_2d'.format(conv_num),
                              summary=summary)

    return out_tensor


def apply_non_lin(non_lin, x, reuse):
    if non_lin:
        if type(non_lin)==str:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(non_lin), reuse)
        else:
            x = non_lin(x)   
            rprint('    Costum non linearity: {}'.format(non_lin), reuse)

    return x


def discriminator(x, params, z=None, reuse=True, scope="discriminator"):
    conv = get_conv(params['is_3d'])

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Discriminator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)
        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)
        if params['cdf']:
            cdf = tf_cdf(x, params['cdf'])
            rprint('    Cdf layer: {}'.format(params['cdf']), reuse)
            rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
            if params['channel_cdf']:
                lst = []
                for i in range(x.shape[-1]):
                    lst.append(tf_cdf(x, params['channel_cdf'],
                                      name="cdf_weight_channel_{}".format(i)))
                    rprint('        Channel Cdf layer: {}'.format(params['cdf']), reuse)
                lst.append(cdf)
                cdf = tf.concat(lst, axis=1)
                rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
            cdf = linear(cdf, 2 * params['cdf'], 'cdf_full', summary=params['summary'])
            cdf = lrelu(cdf)
            rprint('     CDF Full layer with {} outputs'.format(2*params['cdf']), reuse)
            rprint('         Size of the CDF variables: {}'.format(cdf.shape), reuse)
        if params['moment']:
            rprint('    Covariance layer with {} shape'.format(params['moment']), reuse)
            cov = tf_covmat(x, params['moment'])
            rprint('        Layer output {} shape'.format(cov.shape), reuse)
            cov = reshape2d(cov)
            rprint('        Reshape output {} shape'.format(cov.shape), reuse)
            nel = np.prod(params['moment'])**2
            cov = linear(cov, nel, 'cov_full', summary=params['summary'])
            cov = lrelu(cov)
            rprint('     Covariance Full layer with {} outputs'.format(nel), reuse)
            rprint('         Size of the CDF variables: {}'.format(cov.shape), reuse)
            
        for i in range(nconv):
            x = conv(x,
                     nf_out=params['nfilter'][i],
                     shape=params['shape'][i],
                     stride=params['stride'][i],
                     name='{}_conv'.format(i),
                     summary=params['summary'])
            rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = lrelu(x)

        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)

        if z is not None:
            x = tf.concat([x, z], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        if params['cdf']:
            x = tf.concat([x, cdf], axis=1)
            rprint('     Contenate with CDF variables to {}'.format(x.shape), reuse)           
        if params['moment']:
            x = tf.concat([x, cov], axis=1)
            rprint('     Contenate with covairance variables to {}'.format(x.shape), reuse)           

        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)
        if params['minibatch_reg']:
            x = mini_batch_reg(x, n_kernels=150, dim_per_kernel=30)
        x = linear(x, 1, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def generator(x, params, y=None, reuse=True, scope="generator"):
    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])
    with tf.variable_scope(scope, reuse=reuse):
        rprint('Generator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if y is not None:
            x = tf.concat([x, y], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        bs = tf.shape(x)[0]  # Batch size
        # nb pixel
        sx = np.int(
            np.sqrt(np.prod(x.shape.as_list()[1:]) // params['nfilter'][0]))

        if params['is_3d']:
            x = tf.reshape(x, [bs, sx, sx, sx, params['nfilter'][0]], name='vec2img')
        else:
            x = tf.reshape(x, [bs, sx, sx, params['nfilter'][0]], name='vec2img')

        rprint('     Reshape to {}'.format(x.shape), reuse)

        for i in range(nconv):
            sx = sx * params['stride'][i]
            x = deconv(in_tensor=x, 
                       bs=bs, 
                       sx=sx,
                       n_filters=params['nfilter'][i],
                       shape=params['shape'][i],
                       stride=params['stride'][i],
                       summary=params['summary'],
                       conv_num=i,
                       is_3d=params['is_3d'])

            rprint('     {} Deconv layer with {} channels'.format(i+nfull, params['nfilter'][i]), reuse)
            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                    rprint('         Batch norm', reuse)
                x = lrelu(x)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)
        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)

        x = apply_non_lin(params['non_lin'], x, reuse)

        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def generator_up(X, z, params, y=None, reuse=True, scope="generator_up"):
    """
    Arguments
    ---------
    X       : Low sampled image, or None
    z       : Latent variable (same size as X)
    y       : border added a layer param['y_layer']
    """

    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Generator \n'+''.join(['-']*50), reuse)
        if X is not None:
            rprint('     The input X is of size {}'.format(X.shape), reuse)

        rprint('     The input z is of size {}'.format(z.shape), reuse)
        if y is not None:
            rprint('     The input y is of size {}'.format(y.shape), reuse)
        
        ## Increase the dimensionality of the latent variable to that of borders, before adding the border information as channels
        for i in range(nfull):
            z = linear(z,
                       params['full'][i],
                       '{}_full'.format(i),
                       summary=params['summary'])
            z = lrelu(z)
            rprint('     {} Full layer with {} outputs'.format(i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(z.shape), reuse)


        bs = tf.shape(z)[0]  # Batch size
        if X is None:
            sx = y.shape.as_list()[1]
            sy = y.shape.as_list()[2]
        else:
            sx = X.shape.as_list()[1]
            sy = X.shape.as_list()[2]
        
        if params['is_3d']:
            if X is None:
                sz = y.shape.as_list()[3]
            else:
                sz = X.shape.as_list()[3]
            z = tf.reshape(z, [bs, sx, sy, sz, 1], name='vec2img_3d')
        else:
            z = tf.reshape(z, [bs, sx, sy, 1], name='vec2img_2d')
        rprint('     Reshape z to {}'.format(z.shape), reuse)

        if X is None:
            x = z
        else:
            if params['is_3d']:
                x = tf.concat([X, z], axis=4)
            else:
                x = tf.concat([X, z], axis=3)
            rprint('     Concat x and z to {}'.format(x.shape), reuse)

        for i in range(nconv):
            sx = sx * params['stride'][i]

            if (y is not None) and (params['y_layer'] == i):
                rprint('     Merge input y of size{}'.format(y.shape), reuse)

                if params['is_3d']:
                    x = tf.concat([x, y], axis=4)
                else:
                    x = tf.concat([x, y], axis=3)
                rprint('     Concat x and y to {}'.format(x.shape), reuse) 

            x = deconv(in_tensor=x, 
                       bs=bs, 
                       sx=sx,
                       n_filters=params['nfilter'][i],
                       shape=params['shape'][i],
                       stride=params['stride'][i],
                       summary=params['summary'],
                       conv_num=i,
                       is_3d=params['is_3d'])
            rprint('     {} Deconv layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                    rprint('         Batch norm', reuse)

                x = lrelu(x)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)

        x = apply_non_lin(params['non_lin'], x, reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def encoder(x, params, latent_dim, reuse=True, scope="encoder"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Encoder \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        for i in range(nconv):
            x = conv2d(x,
                       nf_out=params['nfilter'][i],
                       shape=params['shape'][i],
                       stride=params['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = lrelu(x)

        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        x = linear(x, latent_dim, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def generator12(x, img, params, reuse=True, scope="generator12"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])

    params_border = params['border']
    assert(len(params_border['stride']) == len(params_border['nfilter'])
           == len(params_border['batch_norm']))
    nconv_border = len(params_border['stride'])
    with tf.variable_scope(scope, reuse=reuse):
        rprint('Border block \n'+''.join(['-']*50), reuse)

        rprint('     BORDER:  The input is of size {}'.format(img.shape), reuse)
        imgt = img
        for i in range(nconv_border):
            imgt = conv2d(imgt,
                          nf_out=params_border['nfilter'][i],
                          shape=params_border['shape'][i],
                          stride=params_border['stride'][i],
                          name='{}_conv'.format(i),
                          summary=params['summary'])
            rprint('     BORDER: {} Conv layer with {} channels'.format(i, params_border['nfilter'][i]), reuse)
            if params_border['batch_norm'][i]:
                imgt = batch_norm(imgt, name='{}_border_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        imgt = reshape2d(imgt, name='border_conv2vec')
        
        st = img.shape.as_list()
        wf = params_border['width_full']
        border = reshape2d(tf.slice(img, [0, st[1]-wf, 0, 0], [-1, wf, st[2], st[3]]), name='border2vec')
        rprint('     BORDER:  Size of the border variables: {}'.format(border.shape), reuse)
        rprint('     BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        rprint('     Latent:  Size of the Z variables: {}'.format(x.shape), reuse)

        x = tf.concat([x, imgt, border], axis=1)
        rprint(''.join(['-']*50)+'\n', reuse)

        x = generator(x, params, reuse=reuse, scope="generator")

        x = tf.concat([img, x], axis=1)

        rprint('     After concatenation: output is of size {}'.format(x.shape), reuse)

        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def one_pixel_mapping(x, n_filters, summary=True, reuse=False):
    """One pixel mapping."""
    rprint('  Begining of one Pixel Mapping '+''.join(['-']*20), reuse)
    xsh = tf.shape(x) 

    rprint('     The input is of size {}'.format(x.shape), reuse)
    x = tf.reshape(x, [xsh[0], prod(x.shape.as_list()[1:]), 1, 1])
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    nconv = len(n_filters)
    for i, n_filter in enumerate(n_filters):
        x = conv2d(x,
                   nf_out=n_filter,
                   shape=[1, 1],
                   stride=1,
                   name='{}_1x1conv'.format(i),
                   summary=summary)

        rprint('     {} 1x1 Conv layer with {} channels'.format(i, n_filter), reuse)    
        x = lrelu(x)
        rprint('         Size of the variables: {}'.format(x.shape), reuse)

    x = conv2d(x,
               nf_out=1,
               shape=[1, 1],
               stride=1,
               name='final_1x1conv',
               summary=summary)
    x = tf.reshape(x, xsh)
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    rprint('  End of one Pixel Mapping '+''.join(['-']*20)+'\n', reuse)
    return x



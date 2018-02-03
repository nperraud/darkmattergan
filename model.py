import tensorflow as tf
import numpy as np
from blocks import *


def rprint(msg, reuse=False):
    if not reuse:
        print(msg)


class WGanModel(object):
    def __init__(self, params, X, z, name='wgan'):
        self.name = name
        self.params = params
        self.G_fake = self.generator(z, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        self._D_loss = D_loss_f - D_loss_r + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r, D_gp)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

class CondWGanModel(object):
    def __init__(self, params, X, z, name='wgan'):
        self.name = name
        self.params = params
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.G_fake = self.generator(z, reuse=False)
        self.D_real = self.discriminator(X, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
        self._D_loss = D_loss_f - D_loss_r + D_gp
        self._G_loss = -D_loss_f
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r, D_gp)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], y=self.y, reuse=reuse)

    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], z=self.y, reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss


class WVeeGanModel(object):
    def __init__(self, params, X, z, name='veegan'):
        self.name = name
        self.params = params
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

        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r, D_gp)

    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, z, reuse):
        return discriminator(X, self.params['discriminator'], z=z, reuse=reuse)

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


class LapGanModel(object):
    def __init__(self, params, X, z, name='lapgan'):
        ''' z must have the same dimension as X'''
        self.name = name
        self.params = params
        self.upsampling = params['generator']['upsampling']
        self.Xs = down_sampler(X, s=self.upsampling)
        inshape = self.Xs.shape.as_list()[1:]
        self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')       
        self.Xsu = up_sampler(self.Xs, s=self.upsampling)
        self.G_fake = self.generator(X=self.y, z=z, reuse=False)
        # self.D_real = self.discriminator(X-self.Xsu, self.Xsu, reuse=False)
        # self.D_fake = self.discriminator(self.G_fake-self.Xsu, self.Xsu, reuse=True)
        self.D_real = -self.discriminator(X, self.Xsu, reuse=False)
        self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)
        D_loss_f = tf.reduce_mean(self.D_fake)
        D_loss_r = tf.reduce_mean(self.D_real)
        gamma_gp = self.params['optimization']['gamma_gp']
        D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
        #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
        self._D_loss = D_loss_f + D_loss_r + D_gp
        self._G_loss = - tf.reduce_mean(self.D_fake)
        wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r, D_gp)
        tf.summary.image("trainingBW/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
        tf.summary.image("trainingBW/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
        tf.summary.image("trainingBW/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])

    def generator(self, X, z, reuse):
        return generator_up(X, z, self.params['generator'], reuse=reuse)

    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss   
       
# class Gan12Model(object):
#     def __init__(self, name='wgan12'):
#         self.name = name
#     def generator(self, z, X, reuse):
#         return generator12(z, X, self.params['generator'], reuse=reuse)
#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)
#     def __call__(self, params, z, X):
#         self.params = params
#         X1, X2 = tf.split(X, 2, axis = params['generator']['border']['axis'])
#         G_fake = self.generator(z, X1, reuse=False)
#         D_real = self.discriminator(X, reuse=False)
#         D_fake = self.discriminator(G_fake, reuse=True)
#         return G_fake, D_real, D_fake


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

def wgan_summaries(D_loss, G_loss, D_loss_f, D_loss_r, D_gp):
    tf.summary.scalar("Disc/Loss", D_loss, collections=["Training"])
    tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
    tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
    tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])
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


def discriminator(x, params, z=None, reuse=True, scope="discriminator"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Discriminator \n------------------------------------------------------------', reuse)
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

        if z is not None:
            x = tf.concat([x, z], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)

        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)
        if params['minibatch_reg']:
            x = mini_batch_reg(x, 16, n_kernels=100, dim_per_kernel=30)
        x = linear(x, 1, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint('------------------------------------------------------------\n', reuse)
    return x


def generator(x, params, y=None, reuse=True, scope="generator"):

    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope):
        rprint('Generator \n------------------------------------------------------------', reuse)
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

        x = tf.reshape(x, [bs, sx, sx, params['nfilter'][0]], name='vec2img')
        rprint('     Reshape to {}'.format(x.shape), reuse)

        for i in range(nconv):
            sx = sx * params['stride'][i]
            x = deconv2d(x,
                         output_shape=[bs, sx, sx, params['nfilter'][i]],
                         shape=params['shape'][i],
                         stride=params['stride'][i],
                         name='{}_deconv'.format(i),
                         summary=params['summary'])
            rprint('     {} Deconv layer with {} channels'.format(i+nfull, params['nfilter'][i]), reuse)
            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                x = lrelu(x)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint('------------------------------------------------------------\n', reuse)
    return x


def generator_up(X, z, params, reuse=True, scope="generator_up"):

    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])

    with tf.variable_scope(scope):
        rprint('Generator \n------------------------------------------------------------', reuse)
        rprint('     The input X is of size {}'.format(X.shape), reuse)

        rprint('     The input z is of size {}'.format(z.shape), reuse)
        bs = tf.shape(X)[0]  # Batch size
        sx = X.shape.as_list()[1]
        sy = X.shape.as_list()[2]
        z = tf.reshape(z, [bs, sx, sy, 1], name='vec2img')        
        rprint('     Reshape z to {}'.format(z.shape), reuse)

        x = tf.concat([X, z], axis=3)
        rprint('     Concat x and z to {}'.format(x.shape), reuse)      

        for i in range(nconv):
            sx = sx * params['stride'][i]
            x = deconv2d(x,
                         output_shape=[bs, sx, sx, params['nfilter'][i]],
                         shape=params['shape'][i],
                         stride=params['stride'][i],
                         name='{}_deconv'.format(i),
                         summary=params['summary'])
            rprint('     {} Deconv layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                x = lrelu(x)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)
        # Xu = up_sampler(X, params['upsampling'])
        # x = x + Xu
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint('------------------------------------------------------------\n', reuse)
    return x


# def generator_up(X, z, params, reuse=True, scope="generator_up"):

#     assert(len(params['stride']) ==
#            len(params['nfilter']) ==
#            len(params['batch_norm'])+1)
#     nconv = len(params['stride'])
#     nfull = len(params['full'])

#     params_encoder = params['encoder']
#     assert(len(params_encoder['stride']) == len(params_encoder['nfilter'])
#            == len(params_encoder['batch_norm']))
#     nconv_encoder = len(params_encoder['stride'])
#     with tf.variable_scope(scope):
#         rprint('Encoder block \n------------------------------------------------------------', reuse)

#         rprint('     ENCODER:  The input is of size {}'.format(X.shape), reuse)
#         imgt = X
#         for i in range(nconv_encoder):
#             imgt = conv2d(imgt,
#                        nf_out=params_encoder['nfilter'][i],
#                        shape=params_encoder['shape'][i],
#                        stride=params_encoder['stride'][i],
#                        name='{}_conv'.format(i),
#                        summary=params['summary'])
#             rprint('     ENCODER: {} Conv layer with {} channels'.format(i, params_encoder['nfilter'][i]), reuse)
#             if params_encoder['batch_norm'][i]:
#                 imgt = batch_norm(imgt, name='{}_border_bn'.format(i), train=True)
#                 rprint('         Batch norm', reuse)
#             rprint('         ENCODER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
#         imgt = reshape2d(imgt, name='border_conv2vec')
        
#         rprint('     ENCODER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
#         rprint('     Latent:  Size of the Z variables: {}'.format(z.shape), reuse)

#         x = tf.concat([z, imgt], axis=1)
#         rprint('------------------------------------------------------------\n', reuse)


#         x  =  generator(x, params, reuse=reuse, scope="generator") 

#         rprint('     Output of the generator {}'.format(x.shape), reuse)
#         rprint('     Adding the interpolated output {}'.format(x.shape), reuse)
        
#         Xu = up_sampler(X, params['upsampling'])
#         if params['non_lin']:
#             non_lin_f = getattr(tf, params['non_lin'])
#             x = non_lin_f(x) + Xu
#             rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)
#             # x = tf.tanh(x + tf.atanh(Xu))
#             # rprint('    Non lienarity: tanh', reuse)
#         else:
#             x = x + Xu

#         rprint('------------------------------------------------------------\n', reuse)

#         return x

def encoder(x, params, latent_dim, reuse=True, scope="encoder"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Encoder \n------------------------------------------------------------', reuse)
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
        rprint('------------------------------------------------------------\n', reuse)
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
    with tf.variable_scope(scope):
        rprint('Border block \n------------------------------------------------------------', reuse)

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
        rprint('------------------------------------------------------------\n', reuse)

        x = generator(x, params, reuse=reuse, scope="generator")

        x = tf.concat([img, x], axis=1)

        rprint('     After concatenation: output is of size {}'.format(x.shape), reuse)

        rprint('------------------------------------------------------------\n', reuse)
    return x


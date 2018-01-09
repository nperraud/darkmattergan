import tensorflow as tf
import numpy as np
from blocks import conv2d, deconv2d, linear, lrelu, batch_norm, reshape2d
from blocks import down_sampler, up_sampler

def rprint(msg, reuse=False):
    if not reuse:
        print(msg)

class gan_model(object):
    def __init__(self, name='gan'):
        self.name = name
    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)
    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)
    def __call__(self, params, z, X):
        self.params = params
        G_fake = self.generator(z, reuse=False)
        D_real = self.discriminator(X, reuse=False)
        D_fake = self.discriminator(G_fake, reuse=True)       
    
        return G_fake, D_real, D_fake

class gan12_model(object):
    def __init__(self, name='wgan12'):
        self.name = name
    def generator(self, z, X, reuse):
        return generator12(z, X, self.params['generator'], reuse=reuse)
    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)
    def __call__(self, params, z, X):
        self.params = params
        X1, X2 = tf.split(X, 2, axis = params['generator']['border']['axis'])
        G_fake = self.generator(z, X1, reuse=False)
        D_real = self.discriminator(X, reuse=False)
        D_fake = self.discriminator(G_fake, reuse=True)       
    
        return G_fake, D_real, D_fake


class veegan_model(object):
    def __init__(self, name='veegan'):
        self.name = name
    def generator(self, z, reuse):
        return generator(z, self.params['generator'], reuse=reuse)
    def discriminator(self, X, z, reuse):
        return discriminator(X, self.params['discriminator'], z=z, reuse=reuse)    
    def encoder(self, X, reuse):
        return encoder(X, self.params['encoder'], self.latent_dim, reuse=reuse)
    def __call__(self, params, z, X):
        self.params = params
        self.latent_dim = params['generator']['latent_dim']
        G_fake = self.generator(z=z, reuse=False)
        z_real = self.encoder(X=X, reuse=False)
        D_real = self.discriminator(X=X, z=z_real, reuse=False)
        D_fake = self.discriminator(X=G_fake, z=z, reuse=True)    
        z_fake = self.encoder(X=G_fake, reuse=True)

        return G_fake, D_real, D_fake, z_real, z_fake


class lapgan(object):
    def __init__(self, name='lapgan'):
        self.name = name
    def generator(self, X, z, reuse):
        return generator_up(X, z, self.params['generator'], reuse=reuse)
    def discriminator(self, X, Xsu, reuse):
        return discriminator(tf.concat([X,Xsu],axis=3), self.params['discriminator'], reuse=reuse)
    def __call__(self, params, z, X):
        self.params = params
        self.upsampling = params['generator']['upsampling']
        Xs = down_sampler(X, s=self.upsampling)
        G_fake = self.generator(X=Xs, z=z, reuse=False)
        Xsu = up_sampler(Xs, s=self.upsampling)
        D_real = self.discriminator(X-Xsu, Xsu, reuse=False)
        D_fake = self.discriminator(G_fake-Xsu, Xsu, reuse=True)       
        
        return G_fake, D_real, D_fake, Xsu


class gan_upsampler(object):
    def __init__(self, name='gan_upsampler'):
        self.name = name
    def generator(self, X, z, reuse):
        return generator_up(X, z, self.params['generator'], reuse=reuse)
    def discriminator(self, X, reuse):
        return discriminator(X, self.params['discriminator'], reuse=reuse)
    def __call__(self, params, z, X):
        self.params = params
        self.upsampling = params['generator']['upsampling']
        Xs = down_sampler(X, s=self.upsampling)
        G_fake = self.generator(X=Xs, z=z, reuse=False)
        G_fake_s = down_sampler(G_fake, s=self.upsampling)
        D_real = self.discriminator(X, reuse=False)
        D_fake = self.discriminator(G_fake, reuse=True)       
        
        return G_fake, D_real, D_fake, Xs, G_fake_s


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

        x = linear(x, 1, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint('------------------------------------------------------------\n', reuse)
    return x


def generator(x, params, reuse=True, scope="generator"):

    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope):
        rprint('Generator \n------------------------------------------------------------', reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
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

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])

    params_encoder = params['encoder']
    assert(len(params_encoder['stride']) == len(params_encoder['nfilter'])
           == len(params_encoder['batch_norm']))
    nconv_encoder = len(params_encoder['stride'])
    with tf.variable_scope(scope):
        rprint('Encoder block \n------------------------------------------------------------', reuse)

        rprint('     ENCODER:  The input is of size {}'.format(X.shape), reuse)
        imgt = X
        for i in range(nconv_encoder):
            imgt = conv2d(imgt,
                       nf_out=params_encoder['nfilter'][i],
                       shape=params_encoder['shape'][i],
                       stride=params_encoder['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     ENCODER: {} Conv layer with {} channels'.format(i, params_encoder['nfilter'][i]), reuse)
            if params_encoder['batch_norm'][i]:
                imgt = batch_norm(imgt, name='{}_border_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         ENCODER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        imgt = reshape2d(imgt, name='border_conv2vec')
        
        rprint('     ENCODER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        rprint('     Latent:  Size of the Z variables: {}'.format(z.shape), reuse)

        x = tf.concat([z, imgt], axis=1)
        rprint('------------------------------------------------------------\n', reuse)



        x  =  generator(x, params, reuse=reuse, scope="generator") 

        rprint('     Output of the generator {}'.format(x.shape), reuse)
        
        x = x + up_sampler(X, params['upsampling'])
        rprint('     Adding the interpolated output {}'.format(x.shape), reuse)

        rprint('------------------------------------------------------------\n', reuse)

        return x

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

        x  =  generator(x, params, reuse=reuse, scope="generator")

        x = tf.concat([img, x], axis=1)

        rprint('     After concatenation: output is of size {}'.format(x.shape), reuse)

        rprint('------------------------------------------------------------\n', reuse)
    return x


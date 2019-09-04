import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from moviepy.video.io.bindings import mplfig_to_npimage

def clusterize_features(features, diff_params, n_components=2):
       
    # Stack features
    nsamples = len(features[0])
    features = np.vstack(features)

    tsne = TSNE(n_components=n_components)
    y = tsne.fit_transform(features)

    fig = plt.figure()
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    for i in range(len(diff_params)):
        current_y = y[(i * nsamples):((i + 1) * nsamples)]
        if n_components == 2:
            plt.scatter(current_y[:, 0], current_y[:, 1], color=('C' + str(i)), label=str(diff_params[i])[:7])
        elif n_components == 3:
            ax.scatter(current_y[:, 0], current_y[:, 1], current_y[:, 2], color=('C' + str(i)), label=str(diff_params[i])[:7])
        else:
            raise ValueError("Impossible to plot " + str(n_components) + " dimensions")
    plt.legend()
    return fig


def visualise_discriminator_filters(wgan, batch, checkpoint=None, top=3, title_func=(lambda x: x), params=None):
    
    if params is None:
        batch = [batch]

    feature_name = lambda x: '_D_conv_activation_' + str(x)
    weight_name = lambda x: 'discriminator/' + str(x) + '_conv/w:0'
    last_best = [0 for i in range(top)]
    layers = range(0, len(wgan.net.params['discriminator']['nfilter']))

    fig_feat, ax_feat = plt.subplots(nrows=len(layers)+1, ncols=top, figsize=(5 * (len(layers)+1), 5 * top))
    fig_weig, ax_weig = plt.subplots(nrows=len(layers), ncols=top, figsize=(5 * len(layers), 5 * top))

    for i in range(top):
        ax_feat[0][i].imshow(batch[0][0])

    # Get all features
    all_features = wgan.get_values_at(batch, [feature_name(l) for l in layers], checkpoint=checkpoint)
        
    for l in layers:

        # Get activations
        # nsamples, x, y, channels
        features = all_features[l]
        
        # Get magnitudes of every filter
        magnitudes = np.zeros(features.shape[3])
        for i in range(len(batch[0])):
            for j in range(features.shape[3]):
                magnitudes[j] = magnitudes[j] + np.linalg.norm(features[i, :, :, j])

        # Get largest activations
        ids = magnitudes.argsort()[-top:][::-1]

        # Plot largest activations
        for i in range(top):
            ax_feat[l+1][i].imshow(features[0, :, :, ids[i]])

        # Plot associated weight
        # x_dim, y_dim, channel_in, channel_out
        # TODO: how to select based on input channel?
        weights = wgan.get_weights(weight_name(l), checkpoint=checkpoint)
        for i in range(len(ids)):
            ax_weig[l][i].imshow(weights[0][:, :, last_best[i], ids[i]])
        last_best = ids

    # Set titles
    if params is not None:
        fig_feat.suptitle(title_func(params), fontsize=16)
        fig_weig.suptitle(title_func(params), fontsize=16)
        fig_feat.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_weig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    return [fig_feat, fig_weig]


def visualise_generator_filters(wgan, batch, checkpoint=None, top=3, title_func=(lambda x: x), params=None):
    
    if params is None:
        batch = [batch]

    feature_name = lambda x: '_G_deconv_activation_' + str(x)
    weight_name = lambda x: 'generator/' + str(x) + '_deconv_2d/w:0'
    last_best = [0 for i in range(top)]
    layers = range(len(wgan.net.params['generator']['nfilter']) - 1, -1, -1)

    fig_feat, ax_feat = plt.subplots(nrows=len(layers)+1, ncols=top, figsize=(5 * (len(layers)+1), 5 * top))
    fig_weig, ax_weig = plt.subplots(nrows=len(layers), ncols=top, figsize=(5 * len(layers), 5 * top))
    
    # Get all features
    # Note need to do it in one step otherwise latent variable is different at every step
    all_features = wgan.get_values_at(batch, ['X_fake'] + [feature_name(l) for l in layers], checkpoint=checkpoint)
    all_features.reverse()
    
    for l in layers:

        # Get activations
        # nsamples, x, y, channels
        features = all_features[l]

        # Get magnitudes of every filter
        magnitudes = np.zeros(features.shape[3])
        for i in range(len(batch[0])):
            for j in range(features.shape[3]):
                magnitudes[j] = magnitudes[j] + np.linalg.norm(features[i, :, :, j])

        # Get largest activations
        ids = magnitudes.argsort()[-top:][::-1]

        # Plot largest activations
        for i in range(top):
            ax_feat[l][i].imshow(features[0, :, :, ids[i]])

        # Plot associated weight
        # x_dim, y_dim, channel_out, channel_in
        # TODO: how to select based on input channel?
        weights = wgan.get_weights(weight_name(l), checkpoint=checkpoint)
        for i in range(len(ids)):
            ax_weig[l][i].imshow(weights[0][:, :, last_best[i], ids[i]])
        last_best = ids
    
    for i in range(top):
        ax_feat[len(layers)][i].imshow(all_features[len(layers)][0, :, :, 0])

    # Set titles
    if params is not None:
        fig_feat.suptitle(title_func(params), fontsize=24)
        fig_weig.suptitle(title_func(params), fontsize=24)
        fig_feat.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_weig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return [fig_feat, fig_weig]


# Returns the make_frame functions to create videos with moviepy
# Each frame consists of the best features / weights for the given params
def make_features_videos(wgan, batch_loader, params, duration, title_func=(lambda x: x), checkpoint=None, top=3, generator=True):

    # Precompute frames
    frames_feat = []
    frames_weig = []
    for p in params:
        
        # Load current batch
        batch = batch_loader(p)

        # Produce figures
        if generator:
            fig_feat, fig_weig = visualise_generator_filters(wgan, batch, checkpoint=checkpoint, top=top, title_func=title_func, params=p)
        else:
            fig_feat, fig_weig = visualise_discriminator_filters(wgan, batch, checkpoint=checkpoint, top=top, title_func=title_func, params=p)

        # Append frames
        frames_feat.append(mplfig_to_npimage(fig_feat))
        frames_weig.append(mplfig_to_npimage(fig_weig))

    # Define make_frame functions
    def make_frame_features(t):
        t = int((len(frames_feat) / duration) * t)
        return frames_feat[t]
    def make_frame_weights(t):
        t = int((len(frames_weig) / duration) * t)
        return frames_weig[t]

    return make_frame_features, make_frame_weights
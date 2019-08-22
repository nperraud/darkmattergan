
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as cm, LogNorm
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import scipy
from scipy import ndimage
import matplotlib.gridspec as gridspec

from gantools import metric
from gantools import utils
import os
import warnings


def draw_images(images,
                nx=1,
                ny=1,
                axes=None,
                *args,
                **kwargs):
    """
    Draw multiple images. This function conveniently draw multiple images side
    by side.

    Parameters
    ----------
    x : List of images
        - Array  [ nx*ny , px, py ]
        - Array  [ nx*ny , px, py , 3]
        - Array  [ nx*ny , px, py , 4]
    nx : number of images to be ploted along the x axis (default = 1)
    ny : number of images to be ploted along the y axis (default = 1)
    px : number of pixel along the x axis (If the images are vectors)
    py : number of pixel along the y axis (If the images are vectors)
    axes : axes

    """
    ndim = len(images.shape)
    nimg = images.shape[0]

    if ndim == 1:
        raise ValueError('Wrong data shape')
    elif ndim == 2:
        images = np.expand_dims(np.expand_dims(images, axis=0), axis=3)
    elif ndim == 3:
        images = np.expand_dims(images, axis=3)
    elif ndim > 4:
        raise ValueError('The input contains too many dimensions')

    px, py, c = images.shape[1:]

    images_tmp = images.reshape([nimg, px, py, c])
    mat = np.zeros([nx * px, ny * py, c])
    for j in range(ny):
        for i in range(nx):
            if i + j * nx >= nimg:
                warnings.warn("Not enough images to tile the entire area!")
                break
            mat[i * px:(i + 1) * px, j * py:(
                j + 1) * py] = images_tmp[i + j * nx, ]
    # make lines to separate the different images
    xx = []
    yy = []
    for j in range(1, ny):
        xx.append([py * j, py * j])
        yy.append([0, nx * px - 1])
    for j in range(1, nx):
        xx.append([0, ny * py - 1])
        yy.append([px * j, px * j])

    if axes is None:
        axes = plt.gca()
    axes.imshow(mat, *args, **kwargs)
    for x, y in zip(xx, yy):
        axes.plot(x, y, color='r', linestyle='-', linewidth=2)
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    return axes


def plot_array_pil(x,
                   pil_object=True,
                   lims=None,
                   log_input=False,
                   log_clip=1e-7):

    norm_x = np.squeeze(x)

    cm = planck_cmap()
    plt.register_cmap(cmap=cm)

    img = Image.fromarray(cm(norm_x, bytes=True))

    if pil_object:
        return img
    else:
        return np.array(img)


# Plot a line with a shade representing either the confidence interval or the standard error
# If confidence is None then the standard error is shown
def plot_with_shade(ax, x, y, label, color, confidence=None, **linestyle):
    transparency = 0.2

    n = y.shape[0]
    y_mean = np.mean(y, axis=0)
    error = (np.var(y, axis=0) / n)**0.5
    if confidence == 'std':
        error = np.std(y, axis=0)
    elif confidence is not None:
        error = error * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    ax.plot(x, y_mean, label=label, color=color, **linestyle)
    ax.fill_between(
        x, y_mean - error, y_mean + error, alpha=transparency, color=color)


def plot_array_plt(x, ax=None, cmap='planck', color='black', simple_k=10):
    if cmap == 'planck':
        cmap = planck_cmap()
        plt.register_cmap(cmap=cmap)

    x = x.reshape(-1)
    size = int(len(x)**0.5)

    log_x = utils.forward_map(x, simple_k)
    lims = [-1, 1]
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.axis([0, size, 0, size])
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')  # labels along the bottom edge are off
    try:
        [i.set_color(color) for i in ax.spines.itervalues()]
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
    except:
        [i.set_color(color) for i in ax.spines.values()]
        [i.set_linewidth(2) for i in ax.spines.values()]

    im = plt.pcolormesh(
        np.reshape(log_x, [size, size]),
        cmap=cmap,
        vmin=lims[0],
        vmax=lims[1],
        edgecolors='face')

    return im


def planck_cmap(ncolors=256):
    """
    Returns a color map similar to the one used for the "Planck CMB Map".
    Parameters
    ----------
    ncolors : int, *optional*
    Number of color segments (default: 256).
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap instance
    Linear segmented color map.
    """
    segmentdata = {
        "red": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                (0.3, 0.00, 0.00), (0.4, 0.00, 0.00), (0.5, 1.00, 1.00),
                (0.6, 1.00, 1.00), (0.7, 1.00, 1.00), (0.8, 0.83, 0.83),
                (0.9, 0.67, 0.67), (1.0, 0.50, 0.50)],
        "green": [(0.0, 0.00, 0.00), (0.1, 0.00, 0.00), (0.2, 0.00, 0.00),
                  (0.3, 0.30, 0.30), (0.4, 0.70, 0.70), (0.5, 1.00, 1.00),
                  (0.6, 0.70, 0.70), (0.7, 0.30, 0.30), (0.8, 0.00, 0.00),
                  (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)],
        "blue": [(0.0, 0.50, 0.50), (0.1, 0.67, 0.67), (0.2, 0.83, 0.83),
                 (0.3, 1.00, 1.00), (0.4, 1.00, 1.00), (0.5, 1.00, 1.00),
                 (0.6, 0.00, 0.00), (0.7, 0.00, 0.00), (0.8, 0.00, 0.00),
                 (0.9, 0.00, 0.00), (1.0, 0.00, 0.00)]
    }
    return cm("Planck-like", segmentdata, N=int(ncolors), gamma=1.0)


def plot_images_psd(images, title, filename=None, sigma_smooth=None):
    my_dpi = 200

    clip_max = 1e10

    images = np.clip(images, -1, clip_max)
    images = utils.makeit_square(images)

    n_rows = len(sigma_smooth)
    # n = n_rows*n_cols
    n = n_rows
    n_cols = 2
    # n_obs = images.shape[0]
    size_image = images.shape[1]
    m = max(5, size_image / my_dpi)
    plt.figure(figsize=(n_cols * m, n_rows * m), dpi=my_dpi)

    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.1, hspace=0.1)

    j = 0
    for i in range(n):
        # fig.add_subplot(gs[i]).set_xlabel(i)
        images1 = ndimage.gaussian_filter(images, sigma=sigma_smooth[i])
        ps_real, k = metric.power_spectrum_batch_phys(X1=images1)

        # PLOTING THE IMAGE
        ax = plt.subplot(gs[j])
        plot_array_plt(
            ndimage.gaussian_filter(images[1], sigma=sigma_smooth[i]),
            ax=ax,
            color='white')
        ax.set_ylabel(
            '$\sigma_{{smooth}} = {}$'.format(sigma_smooth[i]), fontsize=10)
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }

        # PSD
        ax1 = plt.subplot(gs[j + 1])
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        plot_with_shade(
            ax1,
            k,
            ps_real,
            color='b',
            label="Real $\mathcal{F}(X))^2$",
            **linestyle)
        ax1.set_ylim(bottom=0.1)
        if i == 0:
            ax1.title.set_text("2D Power Spectrum\n")
            ax1.title.set_fontsize(11)

        ax1.tick_params(axis='both', which='major', labelsize=10)
        if i == n - 1:
            ax1.set_xlabel("$k$", fontsize=10)
        else:
            ax1.set_xticklabels(())
        j += 2
        # ax1.set_aspect('equal')

    if filename is not None:

        filename = os.path.join('', '{}.png'.format(filename))
        plt.savefig(
            filename, bbox_inches='tight', dpi=my_dpi
        )  # bbox_extra_artists=(txt_top)) #, txt_left))  # Save Image
    plt.show()


# Plot a single curve
def plot_single(x, y, color='b', ax=None, label=None, xscale='linear', yscale='log', xlabel="", ylabel="", title="", shade=False, confidence=None, xlim=None, ylim=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    linestyle = {
        "linewidth": 1,
        "markeredgewidth": 0,
        "markersize": 3,
        "marker": "o",
        "linestyle": "-"
    }
    if shade:
        plot_with_shade(ax, x, y, label=label, color=color, confidence=confidence, **linestyle)
    else:
        if confidence is not None:
            y = np.mean(y, axis=0)
        ax.plot(x, y, label=label, color=color, **linestyle)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, list):
            ylim = ylim[0]
        ax.set_ylim(ylim)
    ax.title.set_text(title + "\n")
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if label:
        ax.legend(fontsize=14, loc=3)


# Comparison plot between two curves real and fake corresponding to axis x
def plot_cmp(x, fake, real=None, xscale='linear', yscale='log', xlabel="", ylabel="", ax=None, title="", shade=False, confidence=None, xlim=None, ylim=None, fractional_difference=False, algorithm='classic', loc=3):
    if ax is None:
        ax = plt.gca()
    if real is None:
        # Plot a line that will be canceled by fake
        # In this way the fractional difference is zero
        plot_single(x, fake, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=False, confidence=confidence, xlim=xlim, ylim=ylim)
    else:
        plot_single(x, real, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    plot_single(x, fake, color='r', ax=ax, label="Fake", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=14, loc=loc)
    if fractional_difference:
        if real is None:
            real = fake
        if shade:
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        plot_fractional_difference(x, real, fake, xscale=xscale, yscale='linear', ax=ax, color='g', ylim=ylim[1] if isinstance(ylim, list) else None, algorithm=algorithm, loc=loc)


def tile_cube_to_2d(cube):
    '''
    cube = [:, :, :]
    arrange cube as tile of squares
    '''
    x_dim = cube.shape[0]
    y_dim = cube.shape[1]
    z_dim = cube.shape[2]
    v_stacks = []
    num = 0
    num_images_in_each_row = utils.num_images_each_row(x_dim)

    for i in range(x_dim//num_images_in_each_row):
        h_stacks = []
        for j in range(num_images_in_each_row): # show 'num_images_in_each_row' squares from the cube in one row
            h_stacks.append(cube[num, :, :])
            num += 1
        v_stacks.append( np.hstack(h_stacks) )

    tile = np.vstack(v_stacks)
    return tile


def tile_and_plot_3d_image(axis, image, **kwargs):
    '''
    Take a 3d cube as input.
    Tile the cube as slices, and display it.
    '''
    tile = tile_cube_to_2d(image)
    #plot = plt.gca()
    axis.imshow(tile, **kwargs)

def cubes_to_animation(cubes, clim=None, figsize=(10,11), title=None, fontsize=24, fps=16, **kwargs):

    if len(cubes.shape)<3:
        cubes = cubes.reshape([1, *cubes.shape])
    if clim is None:
        clim = (np.min(cubes[0]), np.max(cubes[0]))

    fig = plt.figure(figsize=figsize)
    
    nframe = cubes.shape[1]

    def make_frame(t):
        ind = int(round(t*fps))
        plt.cla()
        plt.imshow(cubes[0, ind, :, :], interpolation='none', clim=clim, **kwargs )
        plt.axis('off')
        titlestr = 'Frame no. {}'.format(ind)
        if title:
            titlestr = title + ' - ' + titlestr
        plt.title(titlestr, fontsize=fontsize)

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=nframe/fps)
    plt.clf()

    return animation

def animate_cubes(cubes, output_name='clip', output_format='gif', clim=None, 
    figsize=(10,11), title=None, fontsize=24, fps=16, **kwargs):
    animation = cubes_to_animation(cubes, clim =clim, figsize=figsize, title=title, 
        fontsize=fontsize, fps=fps, **kwargs)
    if output_format=='gif':
        if not (output_name[-4:]=='.gif'):
            output_name += '.gif'
        animation.write_gif(output_name, fps=fps)
        plt.clf()
    elif output_format=='mp4':
        if not (output_name[-4:]=='.mp4'):
            output_name += '.mp4'
        animation.write_videofile(output_name, fps=fps)
        plt.clf()
    # elif output_format=='ipython_display': 
    #     animation.ipython_display(fps=16, loop=True, autoplay=True)
    else:
        raise ValueError('Unknown output_format')

def get_animation(real_cube, fake_cube, real_downsampled=None, clim = None, 
    figsize=(10, 6), fps=16, axis=0, names=['real ', 'real downsampled ', 'fake '], 
    fontsize=20):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions
    Return animation object
    '''
    ind = [0] # has to be a list, as list are mutable
    plt.style.use('dark_background')
    #ax = plt.axes([0,0,1,1], frameon=True)
    #plt.autoscale(tight=False)
    fig = plt.figure(figsize=figsize)

    dim = fake_cube.shape[0]

    if real_downsampled is not None:
        dim_downsampled = real_downsampled.shape[0]
        factor = dim // dim_downsampled
        grid = (1, 3)
        if clim is None:
            cmin = min([np.min(fake_cube[:, :, :]), np.min(real_cube[:, :, :]), np.min(real_downsampled)])
            cmax = max([np.max(fake_cube[:, :, :]), np.max(real_cube[:, :, :]), np.max(real_downsampled)])
            clim =(cmin, cmax)

    else:
        grid = (1, 2)
        if clim is None:
            cmin = min([np.min(fake_cube[:, :, :]), np.min(real_cube[:, :, :])])
            cmax = max([np.max(fake_cube[:, :, :]), np.max(real_cube[:, :, :])])
            clim =(cmin, cmax)

    gridspec.GridSpec(grid[0], grid[1])

    def make_frame(t):

        i = 0
        plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
        plt.imshow(real_cube[ind[0] % dim, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
        plt.title(names[0] + str(dim) + 'x' + str(dim) + 'x' + str(dim), fontsize=fontsize)
        i = i + 1


        if real_downsampled is not None:
            plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
            plt.imshow(real_downsampled[(ind[0] // factor) % dim_downsampled, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
            plt.title(names[1] + str(dim_downsampled) + 'x' + str(dim_downsampled) + 'x' + str(dim_downsampled), fontsize=fontsize)
            i = i + 1


        plt.subplot2grid( grid, (0, i), rowspan=1, colspan=1)
        plt.imshow(fake_cube[ind[0] % dim, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=clim )
        plt.title(names[2] + str(dim) + 'x' + str(dim) + 'x' + str(dim), fontsize=fontsize)
        plt.tight_layout()

        ind[0] += 1
        return mplfig_to_npimage(fig)
    

    animation = VideoClip(make_frame, duration= dim//fps)
    #plt.style.use('default')
    return animation


def save_animation(real_cube, fake_cube, real_downsampled=None, figsize=(10, 6), fps=16, 
    format='gif', output_file_name='test', names=['real ', 'real downsampled ', 'fake '],
    fontsize=20, clim=None):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions, and save it as gif.
    '''
    animation = get_animation(real_cube, fake_cube, real_downsampled, 
        clim=clim, figsize=figsize, fps=fps, names=names, fontsize=fontsize)
    if format == 'gif':
        animation.write_gif(output_file_name + '.gif', fps=fps)
    else:
        animation.write_videofile(output_file_name, fps=fps)
    plt.clf()


# Plot an image
def plot_img(img, x=None, title="", ax=None, cmap=plt.cm.plasma, vmin=None, vmax=None, tick_every=10, colorbar=False, log_norm=False):
    if ax is None:
        ax = plt.gca()
    img = np.squeeze(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=LogNorm() if log_norm else None)
    ax.set_title(title)
    ax.title.set_fontsize(16)
    if x is not None:
        # Define axes
        ticklabels = []
        for i in range(len(x)):
            if i % tick_every == 0:
                ticklabels.append(str(int(round(x[i], 0))))
        ticks = np.linspace(0, len(x) - (len(x) % tick_every), len(ticklabels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
    else:
        ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im


# Plot fractional difference
# if algorithm='classic' then the classic fractional difference abs(real - fake) / real is computed
# if algorithm='relative' then abs(real-fake) / std(real) is computed
def plot_fractional_difference(x, real, fake, xlim=None, ylim=None, xscale="log", yscale="linear", title="", ax=None, color='b', algorithm='classic', loc=3):
    if ax is None:
        ax1 = plt.gca()
    else:
        ax1 = ax.twinx()
    diff = np.abs(real - fake)
    if algorithm == 'classic':
        diff = diff / real
    elif algorithm == 'relative':
        diff = diff / np.std(real, axis=0)
    else:
        raise ValueError("Unknown algorithm name " + str(algorithm))
    plot_single(x, diff, ax=ax1, xscale=xscale, yscale=yscale, title=title, color=color, label='Frac. diff.' if algorithm == 'classic' else 'Rel. diff.', ylim=ylim)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color)

    # Adjust legends
    if ax is not None:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax.get_legend().remove()
        ax1.legend(lines + lines2, labels + labels2, loc=loc, fontsize=14)


def plot_histogram(x, histo, yscale='log', tick_every=10, bar_width=1):
    plt.bar(np.arange(len(histo)), histo, bar_width)
    positions, labels = ([], [])
    for idx in range(len(x)):
        if idx == len(x) - 1:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
        if idx % tick_every == 0:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
    plt.xticks(positions, labels)
    plt.yscale(yscale)


# Plot the scores as a heatmap on the parameter grid
# test_scores: array containing the scores for different parameters
# test_params: array of shape (n, 2) containing the parameters
# thresholds: if None a single heatmap with continuous values is produced
#             if a list is provided a subplot containing different discrete heatmaps 
#             (red points if below threshold and green otherwise) are produced
def plot_heatmap(test_scores, test_params, train_scores=None, train_params=None, vmin=0, vmax=0.25, thresholds=None, labels=['$\Omega_M$', '$\sigma_8$']):
    if thresholds is None:
        plt.scatter(test_params[:, 0], test_params[:, 1], c=test_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k')
        if train_scores is not None and train_params is not None:
            plt.scatter(train_params[:, 0], train_params[:, 1], c=train_scores, vmin=vmin, vmax=vmax, cmap=plt.cm.RdYlGn_r, edgecolor='k', marker='s')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.colorbar()
    else:
        _, ax = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(len(thresholds) * 5, 5))
        for j in range(len(thresholds)):
            if train_params is not None and train_scores is not None:
                for i in range(len(train_params)):
                    ax[j].scatter(train_params[i, 0], train_params[i, 1], c='g' if train_scores[i] <= thresholds[j] else 'r', marker='s')
            for i in range(len(test_params)):
                ax[j].scatter(test_params[i, 0], test_params[i, 1], c='g' if test_scores[i] <= thresholds[j] else 'r')
            ax[j].set_xlabel(labels[0])
            ax[j].set_ylabel(labels[1])
            ax[j].set_title('Threshold: ' + str(thresholds[j]))
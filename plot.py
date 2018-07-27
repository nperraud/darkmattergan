import matplotlib
import socket
if 'nid' in socket.gethostname() or 'lo-' in socket.gethostname():
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap as cm
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from scipy import ndimage
import matplotlib.gridspec as gridspec

import metrics
import utils
import os
import warnings


def draw_images(images,
                nx=1,
                ny=1,
                px=None,
                py=None,
                axes=None,
                *args,
                **kwargs):
    r"""
    Draw multiple images. This function conveniently draw multiple images side
    by side.

    Parameters
    ----------
    x : List of images
        - Matrix [ nx*ny , px*py ]
        - Array  [ nx*ny , px, py ]
    nx : number of images to be ploted along the x axis (default = 1)
    ny : number of images to be ploted along the y axis (default = 1)
    px : number of pixel along the x axis (If the images are vectors)
    py : number of pixel along the y axis (If the images are vectors)
    axes : axes

    """
    ndim = len(images.shape)
    nimg = images.shape[0]

    if ndim == 1:
        raise ValueError('The input seems to contain only one image')
    elif ndim == 2:
        if px and (not py):
            py = int(images.shape[1] / px)
        elif (not px) and py:
            px = int(images.shape[1] / py)
        elif (not px) and (not py):
            raise ValueError('Please specify at least px or py')
        if px * py != images.shape[1]:
            raise ValueError('The sizes do not fit!')
    elif ndim == 3:
        px, py = images.shape[1:]
    else:
        raise ValueError('The input contains to many dimensions')

    images_tmp = images.reshape([nimg, px, py])
    mat = np.zeros([nx * px, ny * py])
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


def plot_with_shade(ax, x, y, label, color, **linestyle):
    transparency = 0.2

    n = y.shape[0]
    y_mean = np.mean(y, axis=0)
    error = (np.var(y, axis=0) / n)**0.5
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
        ps_real, k = metrics.power_spectrum_batch_phys(X1=images1)

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


def get_animation(real_cube, fake_cube, real_downsampled=None, figsize=(4, 8), fps=5, axis=0):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions
    Return animation object
    '''
    ind = [0] # has to be a list, as list are mutable
    fig = plt.figure(figsize=figsize)

    dim = fake_cube.shape[0]

    if real_downsampled is not None:
        dim_downsampled = real_downsampled.shape[0]
        factor = dim // dim_downsampled

    grid = (1, 5)
    gridspec.GridSpec(grid[0], grid[1])

    def make_frame(t):

        cmin = np.min([np.min(fake_cube[:, :, :]), np.min(real_cube[:, :, :])])
        cmax = np.max([np.max(fake_cube[:, :, :]), np.max(real_cube[:, :, :])])


        plt.subplot2grid( grid, (0, 0), rowspan=1, colspan=2)
        plt.imshow(real_cube[ind[0], :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=(cmin, cmax) )
        plt.title('real ' + str(dim) + 'x' + str(dim) + 'x' + str(dim))


        if real_downsampled is not None:
            plt.subplot2grid( grid, (0, 2), rowspan=1, colspan=1)
            plt.imshow(real_downsampled[ind[0] // factor, :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=(np.min(real_downsampled), np.max(real_downsampled)) )
            plt.title('real downsampled '+ str(dim_downsampled) + 'x' + str(dim_downsampled) + 'x' + str(dim_downsampled))


        plt.subplot2grid( grid, (0, 3), rowspan=1, colspan=2)
        plt.imshow(fake_cube[ind[0], :, :], interpolation='nearest', cmap=plt.cm.plasma, clim=(cmin, cmax) )
        plt.title('fake ' + str(dim) + 'x' + str(dim) + 'x' + str(dim))
        plt.tight_layout()

        ind[0] += 1
        return mplfig_to_npimage(fig)
    

    animation = VideoClip(make_frame, duration= dim//fps)
    return animation


def save_animation(real_cube, fake_cube, real_downsampled=None, figsize=(4, 8), fps=5, format='gif', output_file_name='test'):
    '''
    Given real and fake 3d sample, create animation with slices along all 3 dimensions, and save it as gif.
    '''
    animation = get_animation(real_cube, fake_cube, real_downsampled, figsize, fps)
    if format == 'gif':
        animation.write_gif(output_file_name + '.gif', fps=fps)
    else:
        animation.write_videofile(output_file_name, fps=fps)
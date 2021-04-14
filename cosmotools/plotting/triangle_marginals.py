import pylab as plt
import numpy as np
import scipy

def samples_to_probability_2D(data_panel, prob, bins_x, bins_y):

    if prob is None:

        prob2d = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)

    else:

        assert prob.shape[0] == data_panel[0].shape[0]

        hist2d_counts = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)
        hist2d_prob = np.histogram2d(*data_panel, weights=prob, bins=(bins_x, bins_y))[0].T.astype(np.float32)
        prob2d = hist2d_prob/hist2d_counts.astype(np.float)
        prob2d[hist2d_counts==0]=0

    prob2d = prob2d / np.sum(prob2d)
    return prob2d


def samples_to_probability_1D(data, prob, binedges):

    if prob is None:

        prob1D, _ = np.histogram(data, bins=binedges)

    else:

        assert prob.shape[0] == data.shape[0]

        hist_counts, _ = np.histogram(data, bins=binedges)
        hist_prob, _ = np.histogram(data, bins=binedges, weights=prob)
        prob1D = hist_prob/hist_counts.astype(np.float)
        prob1D[hist_counts==0]=0
    
    prob1D = prob1D/np.sum(prob1D)
    return prob1D


def get_kde_grid(data_panel, ranges, columns, i, j, kde_kwargs, prob=None, n_max_points_kde=1000, method='scipy'):

    n_samples = len(data_panel)
    x_ls = np.linspace(*ranges[columns[j]], num=kde_kwargs['n_points'])
    y_ls = np.linspace(*ranges[columns[i]], num=kde_kwargs['n_points'])
    x_grid, y_grid = np.meshgrid(x_ls, y_ls)
    gridpoints = np.vstack([x_grid.ravel(), y_grid.ravel()])


    if method=='scipy':

        if n_samples > n_max_points_kde:
            select = np.random.choice(n_samples, n_max_points_kde)
            data_panel_kde = data_panel[:,select]
        else:
            data_panel_kde = data_panel

        try:
            kernel = scipy.stats.gaussian_kde(data_panel_kde, bw_method='silverman')
        except Exception as err:
            print(data_panel_kde)
            print(err)
            import ipdb, pylab as pl; ipdb.set_trace()

        kde = np.reshape(kernel(gridpoints).T, x_grid.shape)

    elif method=='fastkde':

        from fastkde import fastKDE
        kde, axes = fastKDE.pdf(data_panel[0,:], data_panel[1,:], axes=[x_ls, y_ls])

    elif method=='FFTKDE':

        from KDEpy import FFTKDE
        estimator = FFTKDE(kernel='gaussian')
        est = estimator.fit(data_panel.T)
        kde = est.evaluate(gridpoints.T)

    elif method=='gaussian_mixture':

        from sklearn import mixture
        clf = mixture.GaussianMixture(n_components=20, covariance_type='tied')
        clf.fit(data_panel.T)
        logL = clf.score_samples(gridpoints.T)
        kde = np.exp(logL - np.max(logL)).reshape(x_grid.shape)
        kde = kde/np.sum(kde)

    elif method=='convolve':

        delta_x = x_ls[1] - x_ls[0]
        delta_y = y_ls[1] - y_ls[0]
        bins_x = np.linspace(x_ls[0] - 0.5 * delta_x, x_ls[-1] + 0.5 * delta_x, num=kde_kwargs['n_points'] + 1)
        bins_y = np.linspace(y_ls[0] - 0.5 * delta_y, y_ls[-1] + 0.5 * delta_y, num=kde_kwargs['n_points'] + 1)
        prob2d = samples_to_probability_2D(data_panel, prob, bins_x, bins_y)
        # prob2d = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)
        prob2d = prob2d/np.sum(prob2d)

        from scipy import signal
        from scipy.stats import median_absolute_deviation as mad
        r1 = mad(data_panel[0,:])/(bins_x[1]-bins_x[0])
        r2 = mad(data_panel[1,:])/(bins_y[1]-bins_y[0])
        # sig_pix = np.max([r1,r2])/5
        sig_pix = 1
        kernel = np.outer(signal.gaussian(100, sig_pix), signal.gaussian(100, sig_pix))
        kde = signal.fftconvolve(prob2d, kernel, mode='same')

    kde = kde/np.sum(kde)

    return kde


def get_marginal(data, ranges, columns, i, j, kde_kwargs, prob=None, kde_method='convolve'):


    data_panel = np.vstack((data[columns[j]],
                            data[columns[i]]))

    n_samples = len(data_panel)
    x_ls = np.linspace(*ranges[columns[j]], num=kde_kwargs['n_points'])
    y_ls = np.linspace(*ranges[columns[i]], num=kde_kwargs['n_points'])
    x_grid, y_grid = np.meshgrid(x_ls, y_ls)
    gridpoints = np.vstack([x_grid.ravel(), y_grid.ravel()])

    delta_x = x_ls[1] - x_ls[0]
    delta_y = y_ls[1] - y_ls[0]
    bins_x = np.linspace(x_ls[0] - 0.5 * delta_x, x_ls[-1] + 0.5 * delta_x, num=kde_kwargs['n_points'] + 1)
    bins_y = np.linspace(y_ls[0] - 0.5 * delta_y, y_ls[-1] + 0.5 * delta_y, num=kde_kwargs['n_points'] + 1)
    hist2d = np.histogram2d(*data_panel, bins=(bins_x, bins_y))[0].T.astype(np.float32)
    hist2d = hist2d/np.sum(hist2d)
    kde = get_kde_grid(data_panel, ranges, columns, i, j, kde_kwargs, prob=prob, method=kde_method, n_max_points_kde=1000)

    return kde, hist2d, x_grid, y_grid

def density_image(axc, data, ranges, columns, i, j, fill, color, kde_kwargs, prob=None, kde_method='convolve'):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns 
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    kde_kwargs - dict with kde settings, has to have n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples, in that case samples are treated as grid not a chain
    """
    kde, hist2d, x_grid, y_grid = get_marginal(data, ranges, columns, i, j, kde_kwargs, prob=prob, kde_method='convolve')

    axc.pcolormesh(x_grid, y_grid, kde, cmap=kde_kwargs['cmap'], shading='auto')


def contour_cl(axc, data, ranges, columns, i, j, fill, color, kde_kwargs={}, prob=None, kde_method='convolve', kw_contourf={}):
    """
    axc - axis of the plot
    data - numpy struct array with column data
    ranges - dict of ranges for each column in data
    columns - list of columns 
    i, j - pair of columns to plot
    fill - use filled contour
    color - color for the contour
    kde_kwargs - dict with kde settings, has to have n_points, n_levels_check, levels, defaults below
    prob - if not None, then probability attached to the samples, in that case samples are treated as grid not a chain
    """
    kde_kwargs.setdefault('n_points',200) 
    kde_kwargs.setdefault('n_levels_check',1000) 
    kde_kwargs.setdefault('levels',[0.68, 0.95]) 
    kw_contourf.setdefault('alpha',0.1)

    kde, hist2d, x_grid, y_grid = get_marginal(data, ranges, columns, i, j, kde_kwargs, prob=prob, kde_method='convolve')

    levels_check = np.linspace(0, np.amax(kde), kde_kwargs['n_levels_check'])
    frac_levels = np.zeros_like(levels_check)

    for il, vl in enumerate(levels_check):
        pixels_above_level = kde > vl
        frac_levels[il] = np.sum(pixels_above_level * kde)

    levels_contour = [levels_check[np.argmin(np.fabs(frac_levels - level))] for level in kde_kwargs['levels']][::-1]

    for lvl in levels_contour:
        if fill:
            axc.contourf(x_grid, y_grid, kde, levels=[lvl, np.inf], colors=color, **kw_contourf)
            axc.contour(x_grid, y_grid, kde, levels=[lvl, np.inf], colors=color, alpha=1, linewidths=2)
        else:
            axc.contour(x_grid, y_grid, kde, levels=[lvl, np.inf], colors=color, alpha=1, linewidths=4)

def scatter_density(axc, points1, points2, n_bins=50, lim1=None, lim2=None, norm_cols=False, n_points_scatter=-1, colorbar=False, **kwargs):

    import numpy as np
    if lim1 is None:
        min1 = np.min(points1)
        max1 = np.max(points1)
    else:
        min1 = lim1[0]
        max1 = lim1[1]
    if lim2 is None:
        min2 = np.min(points2)
        max2 = np.max(points2)
    else:
        min2 = lim2[0]
        max2 = lim2[1]

    bins_edges1=np.linspace(min1, max1, n_bins)
    bins_edges2=np.linspace(min2, max2, n_bins)

    hv,bv,_ = np.histogram2d(points1,points2,bins=[bins_edges1, bins_edges2])

    if norm_cols==True:
        hv = hv/np.sum(hv, axis=0)[:,np.newaxis]

    bins_centers1 = (bins_edges1 - (bins_edges1[1]-bins_edges1[0])/2)[1:]
    bins_centers2 = (bins_edges2 - (bins_edges2[1]-bins_edges2[0])/2)[1:]

    from scipy.interpolate import griddata

    select_box = (points1<max1) & (points1>min1) & (points2<max2) & (points2>min2)
    points1_box, points2_box = points1[select_box], points2[select_box]

    x1,x2 = np.meshgrid(bins_centers1, bins_centers2)
    points = np.concatenate([x1.flatten()[:,np.newaxis], x2.flatten()[:,np.newaxis]], axis=1)
    xi = np.concatenate([points1_box[:,np.newaxis], points2_box[:,np.newaxis]],axis=1)


    if lim1 is not None:
        axc.set_xlim(lim1);
    if lim2 is not None:
        axc.set_ylim(lim2)
   

    if n_points_scatter>0:
        select = np.random.choice(len(points1_box), n_points_scatter)
        c = griddata(points, hv.T.flatten(), xi[select,:], method='linear', rescale=True, fill_value=np.min(hv) )
        sc = axc.scatter(points1_box[select], points2_box[select], c=c, **kwargs)
    else:
        c = griddata(points, hv.T.flatten(), xi, method='linear', rescale=True, fill_value=np.min(hv) )
        sorting = np.argsort(c)
        sc = axc.scatter(points1_box[sorting], points2_box[sorting], c=c[sorting],  **kwargs)

    if colorbar:
        plt.gcf().colorbar(sc, ax=axc)


def add_markers(fig, data_markers, tri='lower',scatter_kwargs={}):

    columns = data.dtype.names
    n_dim = len(columns)
    n_box = n_dim+1
    ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    if tri[0]=='l':
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0]=='u':
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception('tri={} should be either l or u'.format(tri))

    for i, j in zip(*tri_indices):

        axc = get_current_ax(ax, tri, i, j)




def plot_triangle_maringals(data, prob=None, func='contour_cl', tri='lower', color='b', ranges={}, ticks={}, n_bins=20, fig=None, fill=True, colors=None, labels=None, alpha_min_contours=None, plot_histograms_1D=True, subplots_kwargs={}, kde_kwargs={}, hist_kwargs={}, axes_kwargs={}, labels_kwargs={}, grid_kwargs={}, scatter_kwargs={}):

    from matplotlib.ticker import FormatStrFormatter

    kde_kwargs.setdefault('n_points', 200)
    kde_kwargs.setdefault('levels', [0.68, 0.95])
    kde_kwargs.setdefault('n_levels_check', 1000)
    kde_kwargs.setdefault('cmap', 'plasma')
    kde_kwargs['levels'].sort()

    grid_kwargs.setdefault('fontsize_ticklabels', 14)
    grid_kwargs.setdefault('tickformat', '{: 0.2e}')

    hist_kwargs.setdefault('histtype', 'step')
    hist_kwargs.setdefault('lw', 4)

    # Get colors
    if colors is None:
        colors = plt.cm.Blues(np.linspace(0.5, 0.8, num=len(kde_kwargs['levels']) + 1))

    columns = data.dtype.names
    n_dim = len(columns)
    n_box = n_dim+1
    n_samples = len(data)

    if tri[0]=='l':
        tri_indices = np.tril_indices(n_dim, k=-1)
    elif tri[0]=='u':
        tri_indices = np.triu_indices(n_dim, k=1)
    else:
        raise Exception('tri={} should be either l or u'.format(tri))

    # Create figure if necessary and get axes
    if fig is None:
        fig, _ = plt.subplots(nrows=n_box, ncols=n_box, figsize=(n_box*4, n_box*4), **subplots_kwargs)
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)
        for axc in ax.ravel():
            axc.axis('off')
    else:
        ax = np.array(fig.get_axes()).reshape(n_box, n_box)

    for c in columns:
        if c not in ranges:
            ranges[c] = (np.amin(data[c]), np.amax(data[c]))
        if c not in ticks:
            ticks[c] = np.linspace(ranges[c][0], ranges[c][1], 5)[1:-1] 

    # Bins for histograms
    hist_binedges = {c: np.linspace(*ranges[c], num=n_bins + 1) for c in columns}
    hist_bincenters = {c: (hist_binedges[c][1:]+hist_binedges[c][:-1])/2 for c in columns}

    def get_current_ax(ax, tri, i, j):

        if tri[0]=='u':
            axc = ax[i, j+1]
        elif tri[0]=='l':
            axc = ax[i+1, j]
        axc.axis('on')
        return axc

    # Plot histograms
    if plot_histograms_1D:
        for i in range(n_dim):

            prob1D = samples_to_probability_1D(data=data[columns[i]], prob=prob, binedges=hist_binedges[columns[i]])

            axc = get_current_ax(ax, tri, i, i)
            axc.plot(hist_bincenters[columns[i]], prob1D, '-', color=color, lw=2)
            axc.fill_between(hist_bincenters[columns[i]], np.zeros_like(prob1D), prob1D, alpha=0.1, color=color)
            axc.set_xlim(ranges[columns[i]])


    # data
    for i, j in zip(*tri_indices):

        axc = get_current_ax(ax, tri, i, j)

        if func=='contour_cl':
            contour_cl(axc, data, ranges, columns, i, j, fill, color, kde_kwargs, prob)
        if func=='density_image':
            density_image(axc, data, ranges, columns, i, j, fill, color, kde_kwargs, prob)
        elif func=='scatter':
            axc.scatter(data[columns[j]], data[columns[i]], c=colors, **scatter_kwargs)
        elif func=='scatter_density':
            scatter_density(axc, points1=data[columns[j]], points2=data[columns[i]], n_bins=n_bins, lim1=ranges[columns[j]], lim2=ranges[columns[i]], norm_cols=False, n_points_scatter=-1)
            
        axc.set_xlim(ranges[columns[j]])
        axc.set_ylim(ranges[columns[i]])
        axc.get_yaxis().set_major_formatter(FormatStrFormatter('%.3e'))
        axc.get_xaxis().set_major_formatter(FormatStrFormatter('%.3e'))      


    # ticks
    n = n_dim-1
        
    # delete all ticks
    for axc in ax.ravel():
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axc.grid(False)


    # ticks
    if tri[0]=='l':
        for i in range(1,n_dim):
            for j in range(0,i):
                axc = get_current_ax(ax, tri, i, j)
                axc.yaxis.tick_left()
                axc.set_yticks(ticks[columns[i]])
        for i in range(1,n_dim):
            for j in range(0,i+1):
                axc = get_current_ax(ax, tri, i, j)
                axc.xaxis.tick_bottom()
                axc.set_xticks(ticks[columns[j]])
    elif tri[0]=='u':
        for i in range(0,n_dim):
            for j in range(i+1, n_dim):
                axc = get_current_ax(ax, tri, i, j)
                axc.yaxis.tick_right()
                axc.set_yticks(ticks[columns[i]])
        for i in range(0,n_dim):
            for j in range(i,n_dim):
                axc = get_current_ax(ax, tri, i, j)
                axc.xaxis.tick_top()
                axc.set_xticks(ticks[columns[j]])       


    def fmt_e(x):
        return grid_kwargs['tickformat'].format(x).replace('e+0', 'e+').replace('e-0', 'e-')



    # ticklabels
    if tri[0]=='l':
        # y tick labels 
        for i in range(1, n_dim):  
            axc = get_current_ax(ax, tri, i, 0)
            ticklabels = [fmt_e(t) for t in ticks[columns[i]]]
            axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')
        # x tick labels
        for i in range(0, n_dim): 
            axc = get_current_ax(ax, tri, n, i)
            ticklabels = [fmt_e(t) for t in ticks[columns[i]]]
            axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')    
    elif tri[0]=='u':
        # y tick labels 
        for i in range(0, n_dim):  
            axc = get_current_ax(ax, tri, i, n)
            ticklabels = [fmt_e(t) for t in ticks[columns[i]]]
            axc.set_yticklabels(ticklabels, rotation=0, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')
        # x tick labels
        for i in range(0, n_dim): 
            axc = get_current_ax(ax, tri, 0, i)
            ticklabels = [fmt_e(t) for t in ticks[columns[i]]]
            axc.set_xticklabels(ticklabels, rotation=90, fontsize=grid_kwargs['fontsize_ticklabels'], family='monospace')    

    # grid
    if tri[0]=='l': 
        for i in range(1,n_dim):
            for j in range(i):  
                axc = get_current_ax(ax, tri, i, j)
                axc.grid(True)
    elif tri[0]=='u':
        for i in range(0,n_dim-1):
            for j in range(i+1,n_dim):  
                axc = get_current_ax(ax, tri, i, j)
                axc.grid(True) 

    # Axes labels
    if labels is None:
        labels = columns


    if tri[0]=='l':
        labelpad = 10
        for i in range(n_dim):
            axc = get_current_ax(ax, tri, i, 0)
            axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
            axc.yaxis.set_label_position("left")
            axc = get_current_ax(ax, tri, n, i)
            axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
            axc.xaxis.set_label_position("bottom")
    elif tri[0]=='u':
        labelpad = 20
        for i in range(n_dim):
            axc = get_current_ax(ax, tri, i, n)
            axc.set_ylabel(labels[i], **labels_kwargs, rotation=90, labelpad=labelpad)
            axc.yaxis.set_label_position("right")
            axc = get_current_ax(ax, tri, 0, i)
            axc.set_xlabel(labels[i], **labels_kwargs, rotation=0, labelpad=labelpad)
            axc.xaxis.set_label_position("top")



    plt.subplots_adjust(hspace=0, wspace=0)

    return fig

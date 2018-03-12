# Generate gaussian based toy problem dataset to test time series GANs
import numpy as np
from scipy.stats import norm
import h5py


def gen_dataset(images_per_time_step=1000, num_time_steps=10, width=64, num_gaussians=42,
                filename=None, point_density_factor=400, size_decay=0.8, offset=10,
                border_width_fraction = 0.1):
    border_width = int(np.round(width * border_width_fraction))
    dataset = np.zeros((num_time_steps, images_per_time_step, width, width))
    num_points = point_density_factor * (width + 2 * border_width) * (width + 2 * border_width) / num_gaussians
    scale = np.power(size_decay, np.arange(num_time_steps) + offset) * width
    pdf_mat = np.zeros((num_time_steps, (width + border_width) * 2 - 1, (width + border_width) * 2 - 1))
    for time_step in range(num_time_steps):
        pdf_vec = norm.pdf(np.arange((width + border_width) * 2 - 1),
                           width - 1 + border_width, scale[time_step])
        pdf_mat[time_step] = np.outer(pdf_vec, pdf_vec)
    for image in range(images_per_time_step):
        for gaussian in range(num_gaussians):
            mu_x = np.random.randint(0, width + 2 * border_width)
            mu_y = np.random.randint(0, width + 2 * border_width)
            for time_step in range(num_time_steps):
                dataset[time_step][image] = dataset[time_step][image]\
                                            + pdf_mat[time_step][mu_x:mu_x+width, mu_y:mu_y+width]
        for time_step in range(num_time_steps):
            dataset[time_step][image] = dataset[time_step][image] * num_points
        if (image + 1) % (images_per_time_step / 20) == 0:
            print('completed {} image series'.format(image + 1))
    #dataset = dataset.astype(int)
    if filename is not None:
        h5f = h5py.File(filename, 'w')
        for time_step in range(num_time_steps):
            h5f.create_dataset(str(time_step), data=dataset[time_step])
        h5f.close()
    return dataset

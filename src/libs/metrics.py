from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def kl(p, q):
    """
    Kullback-Leibler-Divergence
    p, q, must be of shape (n_samples, dimension)
    """
    return entropy(p, q)[0]


def js(p, q):
    """
    Jensen-Shannon-Divergence
    p, q, must be of shape (n_samples, dimension)
    """
    return jensenshannon(p, q)[0]



if __name__ == '__main__':
    from scipy.stats import norm as sci_norm
    from scipy.stats import multivariate_normal as sci_multinorm
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 100)
    dens_1 = sci_norm(4, 0.5).pdf(x)
    dens_2 = sci_norm(6, 0.5).pdf(x)
    dens_3 = sci_norm(8, 0.5).pdf(x)

    compare_list = [
        [dens_1, dens_2],
        [dens_1, dens_3],
        [dens_2, dens_3],
    ]

    distance = 'euclidean'
    # distance = 'sqeuclidean'
    for a, b in compare_list:
        # print('\n{} vs {}'.format(a, b))
        print('\n')
        wasser = ws(a.reshape((-1, 1)), b.reshape((-1, 1)), x.reshape((-1, 1)), distance)
        print("Wasserstein {}".format(wasser))
        jensen = js(a.reshape((-1, 1)), b.reshape((-1, 1)))
        print("Jensen {}".format(jensen))
        kullback = kl(a.reshape((-1, 1)), b.reshape((-1, 1)))
        print("Kullback {}".format(kullback))

    fig, ax = plt.subplots()
    ax.plot(x, dens_1, label='1')
    ax.plot(x, dens_2, label='2')
    ax.plot(x, dens_3, label='3')
    ax.legend()

    fig, ax = plt.subplots()
    x_grid = np.linspace(0, 10, 30)
    y_grid = np.linspace(0, 10, 30)
    Xx, Yy = np.meshgrid(x_grid, y_grid)
    pos_im = np.stack((Xx, Yy), axis=-1)  # (n, n, dim)
    pos_ndim = np.vstack([Xx.ravel(), Yy.ravel()]).T  # (n^2, dim)
    dens_1 = sci_multinorm((4, 3), 0.5).pdf(pos_im)
    dens_2 = sci_multinorm((6, 6), 0.5).pdf(pos_im)
    dens_3 = sci_multinorm((8, 8), 0.5).pdf(pos_im)
    ax.contour(Xx, Yy, dens_1, 10)
    ax.contour(Xx, Yy, dens_2, 10)
    ax.contour(Xx, Yy, dens_3, 10)
    ax.legend()

    compare_list = [
        [dens_1, dens_2],
        [dens_1, dens_3],
        [dens_2, dens_3],
    ]

    for a, b in compare_list:
        print("\n")
        wasser = ws(a.reshape((-1, 1)), b.reshape((-1, 1)), pos_ndim)
        print("Wasserstein {}".format(wasser))
        jensen = js(a.reshape((-1, 1)), b.reshape((-1, 1)))
        print("Jensen {}".format(jensen))
        kullback = kl(a.reshape((-1, 1)), b.reshape((-1, 1)))
        print("Kullback {}".format(kullback))
    plt.show()

from settings import CMAP_OVERLAP, c_vibrant, init

init()

from src.datasets import data_to_torch, random_split
from src.architectures import FNN, FNN_2layer
import argparse
import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import *
from src.datasets import IntroDataset, MySubset
from src.hessian.grads import compute_grads, compute_reinforcing_gradients
from src.utils.general import find_hessian, flatten_grad
from src.utils.plotting import *


def generate_grid_data(min_x, max_x, min_y, max_y, dim_x, dim_y):
    x1 = np.linspace(min_x, max_x, dim_x)
    x2 = np.linspace(min_y, max_y, dim_y)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2, np.vstack([X1.ravel(), X2.ravel()]).T


torch.set_default_dtype(torch.float64)


models = {cls.name: cls for cls in [FNN, FNN_2layer]}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'intro': IntroDataset}

CMAP_GRAD = 'PiYG'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot model for 2D data.')

    parser.add_argument('--config', type=str,
                        help=f'Configuration file from {CONFIG_DIR}.')
    parser.add_argument('--title', type=str, help='Title for the plot.')
    args = parser.parse_args()

    config_file = Path(args.config)
    config_dir = Path('config')

    with open(CONFIG_DIR / config_file, 'r') as f:
        config = json.load(f)

    name = config_file.stem
    dir = config_file.parent

    model_cls = models[config['model']['type']]
    optimizer_cls = optimizers[config['optimizer']['type']]

    dataset = datasets[config['dataset']['type']](**config['dataset']['args'])
    num_classes = dataset.num_classes
    input_size = dataset.input_size
    n = len(dataset)

    train_size = int(config['train_fraction'] * n)
    test_size = int(config['test_fraction'] * n)
    val_size = n - train_size - test_size

    train_data, test_data, val_data = random_split(
        dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    model = model_cls(input_size=input_size,
                      num_classes=num_classes, **config['model']['args'])

    model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))
    criterion = torch.nn.CrossEntropyLoss()

    if test_size == 0.0:
        train_data = test_data = val_data = MySubset(dataset, range(n))
    else:
        train_data, test_data, val_data = random_split(
            dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    figure_path = FIGURE_DIR / 'intro_visuals'
    figure_path.mkdir(parents=True, exist_ok=True)
    import seaborn as sns
    min_x, max_x, dim_x = 0.35, 1.5, 100
    x = np.linspace(min_x, max_x, dim_x).reshape(-1, 1)
    r = model(data_to_torch(x)).detach().numpy()  # math.sqrt(x**2 + y**2)
    grads = compute_grads(model, data_to_torch(x), criterion).T
    min_g, max_g = np.min(grads), np.max(grads)
    clustergrid = sns.clustermap(
        data=grads, row_cluster=True, col_cluster=False)
    plt.clf()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    COLORS = ['royalblue', 'darkorange']
    COLORS = ['dodgerblue', 'orangered']
    COLORS = [c_vibrant['cyan'], c_vibrant['magenta']]

    def prediction(ax):
        ax.plot(x, r[:, 0], label='1', c=COLORS[0])
        ax.plot(x, r[:, 1], label='2', c=COLORS[1])
        ax.set_xlim(min_x, max_x)

        # Plot the maximum betweeen r[:,0] and r[:,1] as a vertical color bar
        # As long as the color does not change, keep going and as soon as it changes, plot the axvspan
        current = np.argmax(r[0, :])
        last_idx = 0
        dones = [False, False]
        for i, x_i in enumerate(x):
            if np.argmax(r[i, :]) != current:
                if not dones[current]:
                    ax.axvspan(x[last_idx], x[i], alpha=0.1,
                               color=COLORS[current], linewidth=0.0, label=current+1)
                    dones[current] = True
                else:
                    ax.axvspan(x[last_idx], x[i], alpha=0.1,
                               color=COLORS[current], linewidth=0.0)
                current = np.argmax(r[i, :])
                last_idx = i
        ax.axvspan(x[last_idx], x[i], alpha=0.2,
                   color=COLORS[current], linewidth=0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        current = np.argmax(r[0, :])
        last_idx = 0
        for i, x_i in enumerate(x):
            if np.argmax(r[i, :]) != current:
                ax.axvline(x[i],  color='black', linestyle=':')
                current = np.argmax(r[i, :])
                last_idx = i

        ax.get_xaxis().set_visible(False)
        ax.set_yticks([0.0])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([' $0$'], fontsize=12)
        ax.set_ylabel('prediction\n'+r'$[f_{\hat{\theta}}(x)]_c$'+'\n')
        ax.legend(title='class $c$', loc='center left',
                  bbox_to_anchor=(1, 0.5))

    # axes[1].get_yaxis().set_visible(False)
    # axes[1].get_xaxis().set_visible(False)

    def data(ax):

        x_data = train_data.all.reshape(-1)
        ax.axhline(0, color='grey', zorder=-6)
        ax.scatter(x_data, np.zeros_like(x_data), c=train_data.labels,
                   cmap=colors.ListedColormap(COLORS), s=65)
        ax.set_xlim(min_x, max_x)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_yticks([0.0])
        ax.set_yticklabels(['y'], fontsize=12)
        # ax.tick_params(axis='y', which='both', length=0)
        ax.set_aspect(1.5)
        ax.set_ylabel('data\n\n')
        ax.set_xlabel('input $x$')

    outputs = model(train_data.all)
    loss = criterion(outputs, train_data.labels)
    hessian = find_hessian(loss, model)
    print("Finding eigenvectors...")
    heigenvalues, heigenvectors = np.linalg.eigh(hessian)
    k = 2
    vectors = [heigenvectors[:, -which_heigenvector]
               for which_heigenvector in range(1, k+1)]
    vectors = [v / np.linalg.norm(v) for v in vectors]
    weights = [heigenvalues[-which_heigenvector]
               for which_heigenvector in range(1, k+1)]

    Y = np.argmax(model(data_to_torch(x)).detach().numpy(), axis=1)
    overlaps = compute_reinforcing_gradients(
        model, data_to_torch(x), vectors, criterion)
    idx = clustergrid.dendrogram_row.reordered_ind

    def align_axis_x(ax, ax_target):
        """Make x-axis of `ax` aligned with `ax_target` in figure"""
        posn_old, posn_target = ax.get_position(), ax_target.get_position()
        ax.set_position([posn_target.x0, posn_old.y0,
                        posn_target.width, posn_old.height])

    def overlaps_plot(ax):

        im = ax.imshow(overlaps, cmap=CMAP_OVERLAP, interpolation='nearest', extent=[
                       min_x, max_x+1/(max_x-min_x)/dim_x, -1, 1], aspect='auto')

        from matplotlib.colors import LogNorm

        # fig.colorbar(im, ax=axes[4])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        current = np.argmax(r[0, :])
        last_idx = 0
        for i, x_i in enumerate(x):
            if np.argmax(r[i, :]) != current:
                ax.axvline(x[i],  color='black', linestyle=':')
                current = np.argmax(r[i, :])
                last_idx = i

        # plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_yticks([0.5, -0.5], fontsize=9)
        ax.set_yticklabels([' $v_1$', ' $v_2$'], fontsize=12)
        ax.get_xaxis().set_visible(False)
        cb = fig.colorbar(im, ax=ax, aspect=10, shrink=0.99)
        cb.outline.set_visible(False)
        ax.set_ylabel('alignment\n'+r'$\mathcal{A}_i(x)$'+'\n')

    def grads_plot(ax):

        im = ax.imshow(grads[idx], cmap=CMAP_GRAD, extent=[
                       min_x, max_x+1/(max_x-min_x)/dim_x, 0, 41], aspect='auto')
        im.set_clim(min_g, max_g)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        last_idx = 0
        current = np.argmax(r[0, :])
        for i, x_i in enumerate(x):
            if np.argmax(r[i, :]) != current:
                ax.axvline(x[i],  color='black', linestyle=':')
                current = np.argmax(r[i, :])
                last_idx = i

        # plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_yticks([0, int(grads.shape[0]/2), grads.shape[0]-1], fontsize=9)
        ax.set_yticklabels(
            [0, int(grads.shape[0]/2), grads.shape[0]-1], fontsize=12)
        ax.get_xaxis().set_visible(False)
        cb = fig.colorbar(im, ax=ax, aspect=10, shrink=0.99)
        cb.outline.set_visible(False)
        ax.set_ylabel('gradients\n'+r'$g_{\hat{\theta}}(x)$'+'\n')

    prediction(axes[0])

    data(axes[3])

    overlaps_plot(axes[2])

    grads_plot(axes[1])

    align_axis_x(axes[3], axes[2])
    align_axis_x(axes[0], axes[2])
    align_axis_x(axes[1], axes[2])

    def hess_vectors(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        im = ax.imshow(np.array(vectors).T[idx], cmap=CMAP_GRAD, aspect=0.05)
        im.set_clim(min_g, max_g)

    plt.savefig(
        figure_path / "data.png", bbox_inches='tight')
    plt.close()

    ############### Hessian Vectors #######
    hess_vectors(plt.gca())
    plt.savefig(figure_path / "hessian_vectors.png")
    plt.close()

    ############### Hessian Spectrum #######
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4, 2/4*3)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(list(reversed(range(1, len(heigenvalues)+1))),
                sorted(heigenvalues), marker='.', c='black')
    plt.xlim(len(heigenvalues)+1, 0)
    plt.xlabel('$i$-th largest eigenvalue', fontsize=9)
    plt.ylabel('$\lambda_i$', fontsize=9)
    ax.tick_params(axis='both', which='both', labelsize=7)
    fig.tight_layout()
    plt.savefig(figure_path / "hessian_spectrum.png")
    plt.close()

    ################ Loss Landscape ################

    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    resolution = 100

    X1, X2, X_mesh = generate_grid_data(
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, dim_x=resolution, dim_y=resolution)
    print(vectors)
    v_1 = vectors[0]
    v_2 = vectors[1]
    v = np.array([v_1, v_2])
    print(v.shape)
    print(X_mesh.shape)

    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    # (100,2) (2,42)
    current_params = flatten_grad(model.parameters())
    outputs = model(train_data.all)
    current_loss = criterion(outputs, train_data.labels).item()

    current_params = current_params.detach().numpy()
    print(current_params.shape)

    P = (X_mesh @ v)
    P += current_params
    L = []
    for p in P:
        pp = torch.from_numpy(p)
        vector_to_parameters(pp, model.parameters())
        outputs = model(train_data.all)
        loss = criterion(outputs, train_data.labels).item()
        L.append(loss)
    L = np.array(L)

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.

    # Plot the surface.
    ax.view_init(12, -60, 0)
    surf = ax.plot_surface(X1, X2, L.reshape(X1.shape), cmap='coolwarm',
                           linewidth=0, antialiased=False)
    ax.scatter([0.0], [0.0], [current_loss], c='red', s=100)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    ax._axis3don = False
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(figure_path / "hessian_loss.png",
                bbox_inches='tight', pad_inches=0)
    plt.close()

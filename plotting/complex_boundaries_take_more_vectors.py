from settings import CMAP_OVERLAP, FIGWIDTH_FULL, init

init()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.architectures import CNN, FNN, FNN_2layer
from src.config import *
from src.datasets import (GaussMixtureDataset, HierachicalGaussMixtureDataset,
                          IrisDataset, data_to_torch, random_split)
from src.hessian.grads import compute_reinforcing_gradients
from src.utils.plotting import *
from src.visualization.utils import *

models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training',
          'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset,
            'iris': IrisDataset, 'hierachical': HierachicalGaussMixtureDataset}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')

    parser.add_argument('--precomputed_hessian',
                        action=argparse.BooleanOptionalAction, help='Use precomputed hessian')
    parser.add_argument(
        '--k', help='Number of eigenvectors to plot', default=7, type=int)
    parser.add_argument('--individual_colorbars', action=argparse.BooleanOptionalAction,
                        help='Use individual colorbars for each plot')
    parser.add_argument('--resolution', type=int,
                        help='How fine the grid should be in the x and y direction.', default=100)
    args = parser.parse_args()

    min_x = min_y = -4
    max_x = max_y = 4

    # Read config file and set model and data
    config_files = [('normal', 'gauss/normal_training.json'),
                    ('adversarial', 'gauss/adversarial_init_training.json'),
                    ('large norm', 'gauss/large_norm_training.json')]

    n_rows = len(config_files)
    n_cols = args.k

    WIDTH_OVERLAP = 1.0 * FIGWIDTH_FULL
    WIDTH_SPECTRUM = 0.3 * FIGWIDTH_FULL
    ratio = np.ones(n_cols+1)
    ratio[-1] = 1.2
    fig, axs = plt.subplots(n_rows, n_cols+1,
                            figsize=(WIDTH_OVERLAP, WIDTH_OVERLAP/(n_cols+1)*n_rows*0.85), gridspec_kw={'width_ratios': ratio})

    cmap_mesh = CMAP_OVERLAP
    all_heigenvalues = []

    for r, (exp_name, cf) in enumerate(config_files):
        config_file = Path(cf)
        config_dir = Path('config')

        with open(CONFIG_DIR / config_file, 'r') as f:
            config = json.load(f)

        name = config_file.stem
        dir = config_file.parent

        figure_path = FIGURE_DIR / 'compelx_boundary' / dir
        figure_path.mkdir(parents=True, exist_ok=True)

        model_cls = models[config['model']['type']]
        optimizer_cls = optimizers[config['optimizer']['type']]

        dataset = datasets[config['dataset']['type']](
            **config['dataset']['args'])
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

        criterion = torch.nn.CrossEntropyLoss()

        X1, X2, X = generate_grid_data(min_x=min_x, max_x=max_x, min_y=min_y,
                                       max_y=max_y, dim_x=args.resolution, dim_y=args.resolution)
        X = data_to_torch(X.astype(np.float32))

        model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))

        outputs = model(train_data.all)
        loss = criterion(outputs, train_data.labels)

        grad_path = GRAD_DIR / dir
        grad_path.mkdir(parents=True, exist_ok=True)
        print('Loading precomputed Hessian')
        hessian = np.load(grad_path / (name + '_hessian.npy'))
        heigenvalues = np.load(grad_path / (name + '_heigenvalues.npy'))
        heigenvectors = np.load(grad_path / (name + '_heigenvectors.npy'))

        vectors = [heigenvectors[:, which_heigenvector]
                   for which_heigenvector in range(heigenvectors.shape[1])]

        Y = np.argmax(model(X).detach().numpy(), axis=1)
        Y = Y.reshape(X1.shape)

        vector_type = 'Top heigenvectors'
        vectors = [heigenvectors[:, -which_heigenvector]
                   for which_heigenvector in range(1, args.k+1)]
        weights = [heigenvalues[-which_heigenvector]
                   for which_heigenvector in range(1, args.k+1)]
        names = [f'$v_{i}$' for i in range(1, args.k+1)]
        eigenvals = [
            f'$\lambda_{i}={heigenvalues[-i]:.3f}$' for i in range(1, args.k+1)]

        weights = np.array(weights)
        weights = weights / weights.sum()
        overlaps = compute_reinforcing_gradients(model, X, vectors)

        axs[r, 0].set_ylabel(exp_name)
        show_decision_boundaries(axs[r, 0], X1, X2, Y.copy(), as_scatter=True)
        show_training_points(axs[r, 0], train_data)
        plt.setp(axs[r, 0].get_xticklabels(), visible=False)
        plt.setp(axs[r, 0].get_yticklabels(), visible=False)
        axs[r, 0].tick_params(axis=u'both', which=u'both', length=0)

        if r == 0:
            axs[r, 0].set_title('decision boundary')

        # Plot the rest of the final plot, the decision boundaries and the overlap
        for i in range(len(vectors)):

            overlap = overlaps[i, :].reshape(X1.shape)
            i = i+1
            im = axs[r, i].pcolormesh(X1, X2, overlap,
                                      cmap=cmap_mesh, shading='nearest')
            if args.individual_colorbars:
                fig.colorbar(im, ax=axs[r, i], shrink=0.75)
            else:
                im.set_clim(-1., 1.)
            show_training_points(axs[r, i], train_data)
            if r == 0:
                axs[r, i].set_title(names[i-1])
            plt.setp(axs[r, i].get_xticklabels(), visible=False)
            plt.setp(axs[r, i].get_yticklabels(), visible=False)
            axs[r, i].tick_params(axis=u'both', which=u'both', length=0)
            axs[r][i].set_xlabel(eigenvals[i-1])

        plt.colorbar(im, ax=axs[r, -1], label='Alignment')

    plt.savefig(
        figure_path / f'ilarge_norm_{name}_{args.k}_normcolor={not args.individual_colorbars}.overlap.png', bbox_inches='tight', dpi=300)
    plt.close()

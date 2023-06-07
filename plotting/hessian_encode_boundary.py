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
from src.datasets import (CircleDataset, GaussCheckerboardLinearClose,
                          GaussCheckerboardNoisyClose, GaussMixtureDataset,
                          HalfMoonDataset, HierachicalGaussMixtureDataset,
                          data_to_torch, random_split)
from src.hessian.grads import compute_reinforcing_gradients
from src.utils.general import find_hessian
from src.utils.plotting import *
from src.visualization.utils import *

models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training',
          'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'circle': CircleDataset, 'half_moon': HalfMoonDataset, 'hierachical': HierachicalGaussMixtureDataset,
            'gauss_checkerboard_noisy_close': GaussCheckerboardNoisyClose, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')
    parser.add_argument('--config', type=str, nargs='+',
                        help=f'Configuration file(s) of model(s) from {CONFIG_DIR}.')
    parser.add_argument('--precomputed_hessian',
                        action=argparse.BooleanOptionalAction, help='Use precomputed hessian')
    parser.add_argument(
        '--overlap_vectors', help='Which type of vectors to compute overlaps with', default='top_heigenvectors')
    parser.add_argument(
        '--k', help='Number of eigenvectors to plot', default=5, type=int)
    parser.add_argument('--individual_colorbars', action=argparse.BooleanOptionalAction,
                        help='Use individual colorbars for each plot')
    parser.add_argument('--hessian_subset', nargs='+', type=int,
                        help='Which subset of class labels to use for the hessian. None is deafault and means using all data.', default=None)
    parser.add_argument('--resolution', type=int,
                        help='How fine the grid should be in the x and y direction.', default=100)
    parser.add_argument('--reparameterize', action=argparse.BooleanOptionalAction,
                        help='Reparameterize the model to make it more sharp.')
    parser.add_argument('--reconstructed_boundary', action=argparse.BooleanOptionalAction,
                        help='Reconstruct boundary using the overlaps weighted by eigenvalues')
    args = parser.parse_args()

    # !! This is not here as a joke, if we save the computed hessian we get problems
    #  with overwriting unless we come up with a nicer naming scheme...
    if args.hessian_subset is None:
        args.hessian_subset = 'all'
    else:
        # ! see above
        assert not args.precomputed_hessian, "Saving Hessian for subset of data is not supported."
    if args.reparameterize:
        # ! see above
        assert not args.precomputed_hessian, "Saving Hessian for reparameterixation is not supported."

    num_rows = len(args.config)
    num_cols = args.k + 1
    if args.reconstructed_boundary:
        num_cols = num_cols + 1

    figure_path = FIGURE_DIR / 'hessian_encode_boundary'
    figure_path.mkdir(parents=True, exist_ok=True)
    dataset_names = ''

    if num_rows == 3:
        ratio = np.ones(num_cols)
        ratio[-1] = 1.2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_rows * num_cols, 2.5 * num_rows),
                                gridspec_kw={'width_ratios': ratio}, squeeze=False)
    else:
        n_cols = num_cols
        n_rows = num_rows
        WIDTH_OVERLAP = 1.0 * FIGWIDTH_FULL
        WIDTH_SPECTRUM = 0.3 * FIGWIDTH_FULL
        ratio = np.ones(n_cols)
        ratio[-1] = 1.2
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False,
                                figsize=(WIDTH_OVERLAP, WIDTH_OVERLAP/(n_cols+1)*n_rows*0.85), gridspec_kw={'width_ratios': ratio})

    cmap_mesh = CMAP_OVERLAP

    for i in range(len(args.config)):
        print(args.config[i])
        config_file = Path(args.config[i])
        config_dir = Path('config')

        with open(CONFIG_DIR / config_file, 'r') as f:
            config_loaded = json.load(f)

        name = config_file.stem
        dir = config_file.parent

        dataset_names += str(dir) + '_'

        model_cls = models[config_loaded['model']['type']]
        optimizer_cls = optimizers[config_loaded['optimizer']['type']]

        dataset = datasets[config_loaded['dataset']['type']](
            **config_loaded['dataset']['args'])
        num_classes = dataset.num_classes
        input_size = dataset.input_size
        n = len(dataset)

        train_size = int(config_loaded['train_fraction'] * n)
        test_size = int(config_loaded['test_fraction'] * n)
        val_size = n - train_size - test_size

        train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size],
                                                       generator=torch.Generator().manual_seed(42))

        model = model_cls(
            input_size=input_size, num_classes=num_classes, **config_loaded['model']['args'])

        if "loss" in config_loaded:
            if config_loaded['loss'] == 'mse':
                criterion = torch.nn.MSELoss()
            elif config_loaded['loss'] == 'nll':
                criterion = torch.nn.NLLLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))

        if args.reparameterize:
            assert isinstance(
                model, FNN_2layer), "Reparameterization only works for FNN_2layers with ReLUs."
            model.reparameterize(0.2, 5.0, 1.0)

        # Calculate Hessian at the minimum along with its spectrum and eigenvectors
        # Run training data through the model
        # only two classes for now
        if args.hessian_subset != 'all':
            for c in args.hessian_subset:
                assert c in train_data.labels, f"Class {c} not in training data."
            mask = train_data.labels == args.hessian_subset[0]
            for c in args.hessian_subset:
                mask = np.logical_or(mask, (train_data.labels == c))
            outputs = model(train_data.all[mask])
            if criterion._get_name() == 'MSELoss':
                loss = criterion(outputs, torch.nn.functional.one_hot(
                    train_data.labels[mask]).to(torch.float32))
            elif criterion._get_name() == 'NLLLoss':
                loss = criterion(outputs, train_data.labels[mask])
            else:
                loss = criterion(outputs, train_data.labels[mask])
        else:
            outputs = model(train_data.all)
            if criterion._get_name() == 'MSELoss':
                loss = criterion(outputs, torch.nn.functional.one_hot(
                    train_data.labels).to(torch.float32))
            elif criterion._get_name() == 'NLLLoss':
                loss = criterion(outputs, train_data.labels)
            else:
                loss = criterion(outputs, train_data.labels)

        # Save Hessian for the analysis with the Hessian-based toolbox
        grad_path = GRAD_DIR / dir
        grad_path.mkdir(parents=True, exist_ok=True)
        if args.precomputed_hessian:
            print('Loading precomputed Hessian')
            hessian = np.load(grad_path / (name + '_hessian.npy'))
            heigenvalues = np.load(grad_path / (name + '_heigenvalues.npy'))
            heigenvectors = np.load(grad_path / (name + '_heigenvectors.npy'))
        else:
            hessian = find_hessian(loss, model)
            print("Finding eigenvectors...")
            heigenvalues, heigenvectors = np.linalg.eigh(hessian)
            if args.hessian_subset != 'all' or args.reparameterize:
                print(
                    'Sorry, not saving hessian of subset or for the reparameterization, this will mess up the saved Hessians so far.')
            else:
                np.save(grad_path / (name + '_hessian'), hessian)
                np.save(grad_path / (name + '_heigenvalues'), heigenvalues)
                np.save(grad_path / (name + '_heigenvectors'), heigenvectors)

        vectors = [heigenvectors[:, -which_heigenvector]
                   for which_heigenvector in range(1, args.k + 1)]
        weights = [heigenvalues[-which_heigenvector]
                   for which_heigenvector in range(1, args.k + 1)]

        # normalize the weighted vectors
        weights = np.array(weights)
        weights_norm = weights / weights.sum()

        if config_loaded['dataset']['type'] == 'gauss_mixtures':
            min_x = min_y = -4
            max_x = max_y = 4
        elif config_loaded['dataset']['type'] == 'circle':
            min_x = min_y = -2
            max_x = max_y = 2
        elif config_loaded['dataset']['type'] == 'half_moon':
            min_x = min_y = -1.7
            max_x = max_y = 2.3
        elif config_loaded['dataset']['type'] == 'hierachical':
            min_x = min_y = -4
            max_x = max_y = 4
        elif config_loaded['dataset']['type'] == 'gauss_checkerboard_noisy_close' or config_loaded['dataset']['type'] == 'gauss_checkerboard_linear_close':
            min_x = -3
            max_x = 3
            min_y = -4
            max_y = 4
        else:
            print('check config file, min max not defined')
            exit()

        X1, X2, X = generate_grid_data(
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, dim_x=100, dim_y=100)
        X = data_to_torch(X.astype(np.float32))
        Y = np.argmax(model(X).detach().numpy(), axis=1)
        Y = Y.reshape(X1.shape)
        overlaps = compute_reinforcing_gradients(model, X, vectors, criterion)

        # Plot first element of the final plot, just the decision boundaries
        show_decision_boundaries(axs[i][0], X1, X2, Y)
        show_training_points(axs[i][0], train_data)
        if i == 0:
            axs[i][0].set_title(f'decision boundary')
            axs[i][0].set_ylabel('$\it{gaussian}$')
        elif i == 1:
            axs[i][0].set_ylabel('$\it{circle}$')
        elif i == 2:
            axs[i][0].set_ylabel('$\it{half-moon}$')
        elif i == 3:
            axs[i][0].set_ylabel('$\it{hierachical}$')
        elif i == 4:
            axs[i][0].set_ylabel('$\it{checkerboard}$')

        axs[i][0].set_yticks([])
        axs[i][0].set_xticks([])

        # Plot the rest of the final plot, the decision boundaries and the overlap
        for j in range(len(vectors)):
            overlap = overlaps[j, :].reshape(X1.shape)
            im = axs[i][j + 1].pcolormesh(X1, X2, overlap,
                                          cmap=cmap_mesh, shading='nearest')
            axs[i][j + 1].set_yticks([])
            axs[i][j + 1].set_xticks([])
            if args.individual_colorbars:
                fig.colorbar(im, ax=axs[i][j + 1], label='Alignment')
            else:
                im.set_clim(-1., 1.)

            # show_decision_boundaries(axs[i+1],X1,X2,Y,as_scatter=True)
            show_training_points(axs[i][j + 1], train_data)
            if i == 0:
                axs[i][j + 1].set_title(f'$v_{j+1}$')
            axs[i][j + 1].set_xlabel(f'$\lambda_{j+1}$ = {weights[j]:.3f}')

        if args.reconstructed_boundary:
            # overlap_composition = np.sum(overlaps, axis=0).reshape(X1.shape) #no weight
            # weight by eigenval
            overlap_composition = (
                overlaps.T @ weights.reshape(-1, 1)).reshape(X1.shape)
            overlap_composition = overlap_composition / \
                np.max(overlap_composition)
            # overlap_composition = (overlaps.T @ weights_norm.reshape(-1, 1)).reshape(X1.shape)  # weight by normalized eigval

            im = axs[i][-1].pcolormesh(X1, X2, overlap_composition,
                                       cmap=cmap_mesh, shading='nearest')
            if args.individual_colorbars:
                fig.colorbar(im, ax=axs[i][-1], label='Alignment')
            else:
                im.set_clim(-1., 1.)

            show_training_points(axs[i][-1], train_data)
            axs[i][-1].set_yticks([])
            axs[i][-1].set_xticks([])
            if i == 0:
                axs[i][-1].set_title(f'reconstructed boundary')

        if not args.individual_colorbars:
            fig.colorbar(im, ax=axs[i][-1], label='Alignment')

    # fig.suptitle(f'{dir}: [{name}] {args.k} {args.overlap_vectors}')
    hs = "".join([str(i) for i in args.hessian_subset])
    plt.savefig(
        figure_path /
        f'{dataset_names}{name}_{args.overlap_vectors}_{args.k}_reconstructed_boundary{args.reconstructed_boundary}.overlap.jpg',
        dpi=600, bbox_inches='tight')
    plt.close()

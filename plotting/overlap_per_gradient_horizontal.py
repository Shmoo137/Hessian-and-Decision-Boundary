from settings import CLASS_COLORS, init, zero

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
                          IrisDataset, MNIST2DDataset, random_split)
from src.hessian.generalization import get_gen_measures
from src.hessian.grads import compute_reinforcing_gradients
from src.utils.general import find_hessian
from src.utils.plotting import *
from src.utils.saving import dump_pickle, load_pickle
from src.visualization.utils import *





models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training',
          'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'iris': IrisDataset, 'mnist2D': MNIST2DDataset, 'hierachical': HierachicalGaussMixtureDataset,
            'circle': CircleDataset, 'half_moon': HalfMoonDataset,
            'gauss_checkerboard_noisy_close': GaussCheckerboardNoisyClose, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}

# for selecting top (factor * num_classes) eigenvalues or eigenvectors
factor = 8

colors = CLASS_COLORS
plt.rcParams["figure.figsize"] = (20, 5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')
    parser.add_argument('--config', type=str, nargs='+',
                        help=f'Configuration file(s) of model(s) from {CONFIG_DIR}.')
    parser.add_argument('--confignames', type=str, nargs='+',
                        help=f'Configuration file(s) of model(s) from {CONFIG_DIR}.')
    parser.add_argument('--precomputed_hessian',
                        action=argparse.BooleanOptionalAction, help='Use precomputed hessian')
    parser.add_argument('--precomputed_overlap',
                        action=argparse.BooleanOptionalAction, help='Use precomputed overlap')
    parser.add_argument('--reparameterize', action=argparse.BooleanOptionalAction,
                        help='Reparameterize the model to make it more sharp.')
    args = parser.parse_args()

    # Define fig objects to iterate over them in a loop
    fig_overlaps, axs_overlaps = plt.subplots(ncols=len(args.config), figsize=[
                                              22/1.5, 2.3/1.5], sharex=True, sharey=True)

    print(axs_overlaps)
    for model_no in range(len(args.config)):
        # Read config file and set model and data
        config_file = Path(args.config[model_no])

        with open(CONFIG_DIR / config_file, 'r') as f:
            config = json.load(f)

        name = config_file.stem
        dir = config_file.parent

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

        model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))

        # Possibly change the parameters of the model
        if args.reparameterize:
            assert isinstance(
                model, FNN_2layer), "Reparameterization only works for FNN_2layers with ReLUs."
            model.reparameterize(0.2, 5.0, 1.0)

        # Calculate Hessian at the minimum along with its spectrum and eigenvectors
        # Run training data through the model
        outputs = model(train_data.all)
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
            print("Computing eigenvalues and eigenvectors")
            heigenvalues, heigenvectors = np.linalg.eigh(hessian)
            np.save(grad_path / (name + '_hessian'), hessian)
            np.save(grad_path / (name + '_heigenvalues'), heigenvalues)
            np.save(grad_path / (name + '_heigenvectors'), heigenvectors)

        # What is encoded by the Hessian eigenvectors corresponding to the largest eigenvalues?
        # Check its overlaps with per-example gradients
        if args.precomputed_overlap:
            print('Loading precomputed overlap')
            overlaps = load_pickle(grad_path / (name + '_overlaps.pkl'))
        else:
            print('Computing overlap')
            vectors = [heigenvectors[:, which_heigenvector]
                       for which_heigenvector in range(heigenvectors.shape[1])]
            overlaps = compute_reinforcing_gradients(
                model, train_data.all, vectors, criterion)
            dump_pickle(overlaps, grad_path / (name + '_overlaps.pkl'))

        # generalization measure weighting by eigenvalue - would not work since overlap values dont mean anything
        # just the sign change is important
        # heigenvalues_normalized = heigenvalues / np.max(heigenvalues)
        # gen_measure = np.sum(np.abs(overlaps).T @ heigenvalues_normalized)
        # print('generalization measure ', gen_measure, gen_measure / train_data.all.__len__())

        # final generalization measure - ratio of number of eigenvectors with non zero overlap

        # We start plotting procedures
        print('Plotting')
        # Set the path to figures
        fig_path = FIGURE_DIR / 'overlap_gradient'

        fig_path.mkdir(parents=True, exist_ok=True)

        print(axs_overlaps)
        # Plot overlaps of all training gradients onto largest Hessian eigenvectors for all setups
        for c in train_data.labels.unique():
            mask = train_data.labels == c
            axs_overlaps[model_no].plot(list(reversed(range(1, factor * num_classes+1))), overlaps[:, mask].mean(1)[-factor * num_classes:],
                                        c=colors[c], label=str(c.item()), lw=3)
        for i in range(train_size):
            axs_overlaps[model_no].plot(list(reversed(range(1, factor * num_classes+1))),
                                        overlaps[:, i][-factor * num_classes:],
                                        c=colors[train_data.labels[i]], alpha=0.1)  # colors[model_no])

            axs_overlaps[model_no].set_ylim(-1, 1)
            axs_overlaps[model_no].set_xlim(factor * num_classes, 1)
            axs_overlaps[model_no].set_title(
                f'{args.confignames[model_no].replace("_"," ")} initialization')

axs_overlaps[0].set_ylabel(f'alignment')
axs_overlaps[1].set_xlabel('$i$-th largest eigenvalue')
axs_overlaps[0].legend(loc='upper left', ncol=4, title='class')

# spectrum of gauss - normal, adv, large norm
data = 'gauss'
grad_path = GRAD_DIR / data
names = ['normal_training', 'adversarial_init_training', 'large_norm_training']
res = {}
for file in names:
    with open(CONFIG_DIR / data / (file + '.json'), 'r') as f:
        config = json.load(f)

    name = file
    dir = CONFIG_DIR / data
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

    heigenvalues = np.load(grad_path / (name + '_heigenvalues.npy'))
    heigenvectors = np.load(grad_path / (name + '_heigenvectors.npy'))

    criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(MODEL_DIR / data / (name + '.pt')))

    norm_gen, norm_tr, norm_l1 = get_gen_measures(
        heigenvectors, heigenvalues, train_data, model, criterion, epsilon=0.04727759086412113)

    res[file] = heigenvalues, norm_gen, norm_tr, norm_l1
    print(res[file])


normal_heigenvalues = res['normal_training'][0]
adversarial_heigenvalues = res['adversarial_init_training'][0]
large_norm_heigenvalues = res['large_norm_training'][0]


def t(x): return r'$\mathcal{G}_\theta=' + \
    f'{x[0]:.3f}$\n$tr(H)={x[1]:.3f}$, $\lambda_1={x[2]:.3f}$'


axs_overlaps[0].text(0.03, 0.35, t(res['normal_training'][1:]),
                     transform=axs_overlaps[0].transAxes, verticalalignment='top')
axs_overlaps[1].text(0.03, 0.35, t(res['adversarial_init_training'][1:]),
                     transform=axs_overlaps[1].transAxes, verticalalignment='top')
axs_overlaps[2].text(0.03, 0.35, t(res['large_norm_training'][1:]),
                     transform=axs_overlaps[2].transAxes, verticalalignment='top')

fig_overlaps.savefig(
    fig_path / f'{name}.Hessian_overlaps{".sharp" if args.reparameterize else ""}.png', bbox_inches='tight')

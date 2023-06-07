from settings import CMAP_OVERLAP, FIGWIDTH_COLUMN, INIT_COLORS, init, zero

init()

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.architectures import CNN, FNN, FNN_2layer
from src.config import *
from src.datasets import (CircleDataset, GaussCheckerboardLinearClose,
                          GaussCheckerboardNoisyClose, GaussMixtureDataset,
                          HalfMoonDataset, HierachicalGaussMixtureDataset,
                          IrisDataset, random_split)
from src.hessian.generalization import generalization_measure
from src.hessian.grads import compute_reinforcing_gradients
from src.utils.plotting import *
from src.visualization.utils import *


def get_gen_measures(heigenvectors, heigenvalues, train_data, model, criterion):
    vectors = [heigenvectors[:, which_heigenvector]
               for which_heigenvector in range(heigenvectors.shape[1])]
    overlaps = compute_reinforcing_gradients(
        model, train_data.all, vectors, criterion)
    gen = generalization_measure(overlaps, zero=zero)
    tr = sum(heigenvalues)
    l1 = heigenvalues[-1]

    return (gen, tr, l1)


models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training',
          'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'iris': IrisDataset, 'hierachical': HierachicalGaussMixtureDataset,
            'circle': CircleDataset, 'half_moon': HalfMoonDataset, 'gauss_checkerboard_noisy_close': GaussCheckerboardNoisyClose,
            'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')
    parser.add_argument('--dataset', type=str, default='gauss')
    args = parser.parse_args()

    figure_path = FIGURE_DIR / 'generalization_histogram'
    figure_path.mkdir(parents=True, exist_ok=True)
    dataset_names = ''

    num_rows = 3
    num_cols = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(
        0.25*FIGWIDTH_COLUMN*2.2/1.5, FIGWIDTH_COLUMN*1.3/1.5))

    cmap_mesh = CMAP_OVERLAP

    # spectrum of gauss - normal, adv, large norm
    data = args.dataset
    grad_path = GRAD_DIR / data
    names = ['normal_training',
             'adversarial_init_training', 'large_norm_training']
    res = {}
    for file in names:
        with open(CONFIG_DIR / data / (file + '.json'), 'r') as f:
            config = json.load(f)

        name = file
        dir = CONFIG_DIR / data
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

        heigenvalues = np.load(grad_path / (name + '_heigenvalues.npy'))
        heigenvectors = np.load(grad_path / (name + '_heigenvectors.npy'))

        criterion = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(MODEL_DIR / data / (name + '.pt')))

        norm_gen, norm_tr, norm_l1 = get_gen_measures(
            heigenvectors, heigenvalues, train_data, model, criterion)

        res[file] = heigenvalues, norm_gen, norm_tr, norm_l1
        print(res[file])

    normal_heigenvalues = res['normal_training'][0]
    adversarial_heigenvalues = res['adversarial_init_training'][0]
    large_norm_heigenvalues = res['large_norm_training'][0]

    nbins = 200

    n, bins, patches = axs[0].hist(normal_heigenvalues, bins=nbins, density=True,
                                   edgecolor=INIT_COLORS['normal'], color=INIT_COLORS['normal'], linewidth=1.5)
    for data, b in zip(n, bins):
        if 0 < data < 3 and b > 0.01:
            axs[0].scatter([b], [1], marker='x', color='black', s=10)

    def t(x): return r'$\mathcal{G}_\theta=' + \
        f'{x[0]:.3f}$\n$tr(H)={x[1]:.3f}$\n$\lambda_1={x[2]:.3f}$'

    axs[0].set_title('eigenspectrum')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('$P(\lambda_i)$')
    axs[1].set_ylabel('$P(\lambda_i)$')
    axs[2].set_ylabel('$P(\lambda_i)$')
    nn, binsbins, patches = axs[1].hist(adversarial_heigenvalues, bins=nbins, density=True,
                                        edgecolor=INIT_COLORS['adversarial'], color=INIT_COLORS['adversarial'], linewidth=1.5)

    axs[1].set_yscale('log')
    for data, b in zip(nn, binsbins):
        if 0 < data < 200 and b > 0.0003:  # 0.0001:
            axs[1].scatter([b], [27], marker='x', color='black', s=10)

    n, bins, patches = axs[2].hist(large_norm_heigenvalues, bins=nbins, density=True,
                                   edgecolor=INIT_COLORS['large norm'], color=INIT_COLORS['large norm'], linewidth=1.5)
    xx = 0
    for data, b in zip(n, bins):
        if 0 < data < 200 and b > 0.00025:
            if xx == 0:
                axs[2].scatter([b], [13], marker='x',
                               color='black', s=10, label='outlier')
                xx = 1
            axs[2].scatter([b], [13], marker='x', color='black', s=10)

    """
    axs[0].set_xlim(10e-6,10e-1)
    axs[1].set_xlim(10e-6,10e-1)
    axs[2].set_xlim(10e-6,10e-1)
    
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')
    """
    axs[2].set_yscale('log')

    axs[2].set_xlabel('$\lambda_i$')

    print(figure_path / f'eigenvalues_hist_{nbins}_gauss_vertical.jpg')

    plt.savefig(
        figure_path / f'eigenvalues_hist_{nbins}_gauss_vertical.jpg',
        dpi=600, bbox_inches='tight')
    plt.close()

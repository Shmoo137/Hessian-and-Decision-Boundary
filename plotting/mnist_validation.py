# %%
from src.visualization.utils import *
from src.datasets import random_split
from src.datasets import GaussMixtureDataset, CircleDataset, HalfMoonDataset, data_to_torch, MNIST2DDataset
from src.hessian.grads import compute_reinforcing_gradients
from src.utils.general import find_hessian
from src.architectures import FNN, FNN_2layer, CNN
from src.utils.plotting import *
from src.config import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import json
import argparse
from settings import init, CMAP_OVERLAP, CMAP_MNIST, INIT_COLORS
from sklearn.manifold import TSNE
import seaborn as sns

init()


models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training',
          'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'mnist2D': MNIST2DDataset, 'gauss_mixtures': GaussMixtureDataset, 'circle': CircleDataset, 'half_moon': HalfMoonDataset}


"""
Plot for showing that really the top eigenvectors matter.
"""


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
    num_cols = args.k + 3
    if args.reconstructed_boundary:
        num_cols = num_cols + 1

    dataset_names = ''

    ratio = np.ones(num_cols)
    ratio[-3] = 1.2
    ratio[-2] = 0.1
    ratio[-1] = 1.5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_rows * num_cols, 2 * num_rows),
                            gridspec_kw={'width_ratios': ratio})
    cmap_mesh = CMAP_OVERLAP

    for i in range(len(args.config)):
        print(args.config[i])
        config_file = Path(args.config[i])
        config_dir = Path('config')

        with open(CONFIG_DIR / config_file, 'r') as f:
            config_loaded = json.load(f)

        name = config_file.stem
        name_truncated = name.rsplit('_',1)[0]
        dir = config_file.parent

        figure_path = FIGURE_DIR / 'real_data' / dir
        figure_path.mkdir(parents=True, exist_ok=True)

        model_cls = models[config_loaded['model']['type']]
        optimizer_cls = optimizers[config_loaded['optimizer']['type']]

        dataset = datasets[config_loaded['dataset']['type']](**config_loaded['dataset']['args'])
        num_classes = dataset.num_classes
        input_size = dataset.input_size
        class_list = config_loaded['dataset']['args']['class_list']
        train_seed = (args.config[i].split('.')[0]).split('_')[-1]
        dataset_names += ''.join(map(str,class_list)) + '_'+train_seed+'_'
        n = len(dataset)

        train_size = int(config_loaded['train_fraction'] * n)
        test_size = int(config_loaded['test_fraction'] * n)
        val_size = n - train_size - test_size

        train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size],
                                                       generator=torch.Generator().manual_seed(42))

        model = model_cls(input_size=input_size, num_classes=num_classes, **config_loaded['model']['args'])

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
                loss = criterion(outputs, torch.nn.functional.one_hot(train_data.labels[mask]).to(torch.float32))
            elif criterion._get_name() == 'NLLLoss':
                loss = criterion(outputs, train_data.labels[mask])
            else:
                loss = criterion(outputs, train_data.labels[mask])
        else:
            outputs = model(train_data.all)
            if criterion._get_name() == 'MSELoss':
                loss = criterion(outputs, torch.nn.functional.one_hot(train_data.labels).to(torch.float32))
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

        vectors = [heigenvectors[:, -which_heigenvector] for which_heigenvector in range(1, args.k + 1)]
        weights = [heigenvalues[-which_heigenvector] for which_heigenvector in range(1, args.k + 1)]

        # normalize the weighted vectors
        weights = np.array(weights)
        weights_norm = weights / weights.sum()

        # get TSNE components of the train data
        tsne = TSNE(n_components=2, random_state=42)
        tsne_res = tsne.fit_transform(train_data.all.reshape(train_data.all.shape[0], -1))

        # relabel to actual class for proper legend
        labels = train_data.labels
        y_idx_list = [torch.where(labels == k)[0] for k in np.arange(num_classes)]
        for k in range(len(y_idx_list)):
            labels[y_idx_list[k]] = class_list[k]

        X = train_data.all.reshape(train_data.all.shape[0], 1, train_data.all.shape[1], train_data.all.shape[2],
                                   train_data.all.shape[3])
        overlaps = compute_reinforcing_gradients(model, X, vectors, criterion, mnist=True)

        # plot TSNE visualization of the training data
        all_colors = CMAP_MNIST
        colors = [all_colors[c] for c in class_list]
        im = sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=labels, palette=sns.color_palette(colors),
                             legend='full', ax=axs[i][0])
        if i == 0:
            axs[i][0].set_title(f't-SNE')

        norm = plt.Normalize(np.min(overlaps), np.max(overlaps))
        sm = plt.cm.ScalarMappable(cmap=CMAP_OVERLAP, norm=norm)

        # Plot the rest of the final plot, the decision boundaries and the overlap
        for j in range(len(vectors)):
            im = sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], c=overlaps[j, :], cmap=cmap_mesh, norm=norm, ax=axs[i][j+1])
            axs[i][j + 1].set_yticks([])
            axs[i][j + 1].set_xticks([])

            if i == 0:
                axs[i][j + 1].set_title(f'$v_{j+1}$')
            axs[i][j + 1].set_xlabel(f'$\lambda_{j+1}= {weights[j]:.3f}$')

        if args.reconstructed_boundary:
            overlap_composition = (overlaps.T @ weights.reshape(-1,1)) # weight by eigenval
            overlap_composition = overlap_composition/np.max(overlap_composition)

            im = sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], c=overlap_composition, cmap=cmap_mesh, norm=norm, ax=axs[i][-1])

            axs[i][-1].set_yticks([])
            axs[i][-1].set_xticks([])
            if i == 0:
                axs[i][-1].set_title(f'reconstructed boundary')

        if not args.individual_colorbars:
            fig.colorbar(mappable=sm, ax=axs[i][-3], label='Alignment')

        nbins = 200

        cls = ''.join(map(str,class_list))
        train_type = str(name_truncated.split('_')[0])
        axs[i][0].set_ylabel(train_type)
        axs[0][-1].set_title(f'eigenspectrum')
        axs[i][-1].set_ylabel(f'$P(\lambda_i)$')
        axs[-1][-1].set_xlabel(f'$\lambda_i$')

        if train_type == 'normal':
            n, bins, patches = axs[i][-1].hist(heigenvalues, bins=nbins, density=True,
                                              edgecolor=INIT_COLORS[train_type],
                                              color=INIT_COLORS[train_type], linewidth=1.5)
            for data, b in zip(n, bins):
                if (cls == '017' and 0 < data < 3 and b > 0.5) or \
                        (cls == '179' and 0 < data < 3 and b > 1) or \
                        (cls == '0179' and 0 < data < 3 and b > 1.5) or \
                        (cls == '1379' and 0 < data < 3 and b > 1):
                    axs[i][-1].scatter([b], [0.1], marker='x', color='black', s=10)

            axs[i][-1].set_yscale('log')

        elif train_type == 'adversarial':
            nn, binsbins, patches = axs[i][-1].hist(heigenvalues, bins=nbins, density=True,
                                                edgecolor=INIT_COLORS[train_type], color=INIT_COLORS[train_type],
                                                linewidth=1.5)
            for data, b in zip(nn, binsbins):
                if (cls == '017' and 0 < data < 3 and b > 0.5) or \
                        (cls == '179' and 0 < data < 3 and b > 0.25) or \
                        (cls == '0179' and 0 < data < 3 and b > 0.15) or \
                        (cls == '1379' and 0 < data < 3 and b > 0.1):
                    axs[i][-1].scatter([b], [0.5], marker='x', color='black', s=10)
            axs[i][-1].set_yscale('log')

        for ax in axs[:, -2]:
            ax.axis("off")

    hs = "".join([str(i) for i in args.hessian_subset])
    fig.tight_layout()
    plt.savefig(
        figure_path / f'{dataset_names}{args.overlap_vectors}_{args.k}_normcolor={not args.individual_colorbars}_hessian_subset={hs}_reparameterized={args.reparameterize}_reconstructed_boundary{args.reconstructed_boundary}.overlap.png',
        dpi=600)
    plt.show()
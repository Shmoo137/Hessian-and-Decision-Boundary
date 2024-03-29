import argparse
import json
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from hessian_eigenthings import compute_hessian_eigenthings

#torch.set_default_dtype(torch.float64)

from src.utils.plotting import *
from src.utils.saving import dump_pickle, load_pickle
from src.config import *
from src.architectures import *
from src.utils.general import find_hessian
from src.hessian.grads import compute_reinforcing_gradients, compute_grad, compute_logit_grads
from src.datasets import *

models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN, LeNet5, ResNet18]}
setups = {'normal': 'normal_training', 'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'iris': IrisDataset,
            'mnist2D': MNIST2DDataset, 'cifar10': CIFAR10, 'hierachical': HierachicalGaussMixtureDataset,
            'circle': CircleDataset, 'half_moon': HalfMoonDataset, 'intro': IntroDataset,
            'gauss_checkerboard_noisy_close': GaussCheckerboardNoisyClose, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}

factor = 10                                 # for selecting top (factor * num_classes) eigenvalues or eigenvectors

""" To compare multiple models, use: --config configA configB configC... 
When comparing, figs get saved to ./Hessian_comparison, so be careful if you're comparing multiple setups """

colors = [c_vibrant['blue'], c_vibrant['cyan'], c_vibrant['magenta'], c_vibrant['orange']]
plt.rcParams["figure.figsize"] = (20, 5)

plot_spectra = False
plot_eigenvectors = False
plot_overlaps = False
plot_coherence = False
plot_logit_coherence = False

zero = 1e-6

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')
    parser.add_argument('--config', type=str, nargs='+', help=f'Configuration file(s) of model(s) from {CONFIG_DIR}.')
    parser.add_argument('--num_eigenthings', type=int, default = 0, help=f'Number of the top heigenvalues and heigenvectors (if approx).')
    parser.add_argument('--precomputed_hessian', action=argparse.BooleanOptionalAction, help='Use precomputed hessian')
    parser.add_argument('--precomputed_overlap', action=argparse.BooleanOptionalAction, help='Use precomputed overlap')
    parser.add_argument('--precomputed_logit_grads', action=argparse.BooleanOptionalAction, help='Use precomputed logit gradients')
    parser.add_argument('--approximate_hessian', action=argparse.BooleanOptionalAction, help='Approximate the hessian')
    parser.add_argument('--reparameterize', action=argparse.BooleanOptionalAction, help='Reparameterize the model to make it more sharp.')
    args = parser.parse_args()
    #assert not args.precomputed_hessian or args.precomputed_overlap, "Cannot load precomputed overlap without precomputed hessian"
    

    ## Define fig objects to iterate over them in a loop
    if plot_spectra is True:
        fig_spectra, ax_spectra = plt.subplots(nrows=2, figsize=[20, 5])
    
    if plot_eigenvectors is True:
        how_many_largest_eigenvectors = 3
        fig_eigenvectors, axs_eigenvectors = plt.subplots(nrows=how_many_largest_eigenvectors, ncols=len(args.config), figsize=[20, 15], squeeze = False)
    
    if plot_overlaps is True:
        fig_overlaps, axs_overlaps = plt.subplots(nrows=2, ncols=len(args.config), figsize=[20, 10], squeeze = False)
    
    if plot_coherence is True:
        fig_normalized_coherence, axs_normalized_coherence = plt.subplots(nrows=1, ncols=len(args.config), figsize=[20, 10], squeeze = False)

    if plot_logit_coherence is True:
        fig_logit_normalized_coherence, axs_logit_normalized_coherence = plt.subplots(nrows=1, ncols=len(args.config), figsize=[20, 10], squeeze = False)

    for model_no in range(len(args.config)):
        print("Analyzing model no.", model_no + 1, "/", len(args.config))
        ## Read config file and set model and data
        config_file = Path(args.config[model_no])

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

        train_size =  int(config['train_fraction'] * n)
        test_size = int(config['test_fraction'] * n)
        val_size = n - train_size - test_size

        train_data, test_data, val_data = random_split(dataset, [train_size,test_size,val_size], generator=torch.Generator().manual_seed(42))

        model = model_cls(input_size=input_size, num_classes=num_classes, **config['model']['args'])

        criterion = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))

        train_set = torch.utils.data.TensorDataset(
            train_data.all.float(),
            train_data.labels)
        
        train_iterator = torch.utils.data.DataLoader(train_set,
                                                     shuffle=False,
                                                     batch_size=config['trainer']['batch_size'])

        # Possibly change the parameters of the model
        if args.reparameterize:
            assert isinstance(model, FNN_2layer), "Reparameterization only works for FNN_2layers with ReLUs."
            model.reparameterize(0.2,5.0,1.0)

        ## Calculate Hessian at the minimum along with its spectrum and eigenvectors
        # Run training data through the model
        outputs = model(train_data.all)
        loss = criterion(outputs, train_data.labels)
        
        # Save Hessian for the analysis with the Hessian-based toolbox
        grad_path = GRAD_DIR / dir
        grad_path.mkdir(parents=True, exist_ok=True)
        if args.precomputed_hessian:
            print('Loading precomputed Hessian')
            hessian = np.load(grad_path /  (name + '_hessian.npy'))
            heigenvalues = np.load(grad_path /  (name + '_heigenvalues.npy'))
            heigenvectors = np.load(grad_path /  (name + '_heigenvectors.npy'))
        elif args.approximate_hessian:
            heigenvalues, heigenvectors = compute_hessian_eigenthings(model, train_iterator, criterion, args.num_eigenthings, use_gpu = False)
            np.save(grad_path /  (name + '_heigenvalues'), heigenvalues)
            np.save(grad_path /  (name + '_heigenvectors'), heigenvectors)
        else:
            hessian = find_hessian(loss, model)
            print("Computing eigenvalues and eigenvectors")
            heigenvalues, heigenvectors = np.linalg.eigh(hessian)
            np.save(grad_path /  (name + '_hessian'), hessian)
            np.save(grad_path /  (name + '_heigenvalues'), heigenvalues)
            np.save(grad_path /  (name + '_heigenvectors'), heigenvectors)
        
        ## What is encoded by the Hessian eigenvectors corresponding to the largest eigenvalues?
        # Check its overlaps with per-example gradients
        if args.precomputed_overlap:
            print('Loading precomputed overlap')
            overlaps = load_pickle(grad_path /  (name + '_overlaps.pkl'))
        else:
            print('Computing overlap')
            vectors = [heigenvectors[:, which_heigenvector] for which_heigenvector in range(heigenvectors.shape[1])]
            mnist = False
            if config['dataset']['type'] == 'mnist2D' or config['dataset']['type'] == 'cifar10':
                mnist = True
            overlaps = compute_reinforcing_gradients(model, train_data.all, vectors, criterion, mnist=mnist)
            dump_pickle(overlaps,grad_path /  (name + '_overlaps.pkl'))

        ## Compute logit gradients to check how they cluster
        if args.precomputed_logit_grads:
            print('Loading precomputed logit gradients')
            logit_gradients = load_pickle(grad_path /  (name + '_logit_gradients.pkl'))
        else:
            print('Computing logit gradients')
            mnist = False
            if config['dataset']['type'] == 'mnist2D' or config['dataset']['type'] == 'cifar10':
                mnist = True
            logit_gradients = compute_logit_grads(model, train_data, mnist=mnist, sort=True)
            dump_pickle(logit_gradients, grad_path /  (name + '_logit_gradients.pkl'))

        # generalization measure weighting by eigenvalue - would not work since overlap values dont mean anything
        # just the sign change is important
        # heigenvalues_normalized = heigenvalues / np.max(heigenvalues)
        # gen_measure = np.sum(np.abs(overlaps).T @ heigenvalues_normalized)
        # print('generalization measure ', gen_measure, gen_measure / train_data.all.__len__())

        # final generalization measure - ratio of number of eigenvectors with non zero overlap
        overlap_mean = np.mean(np.abs(overlaps), axis=1) # can use this for plotting if needed
        overlap_spread = np.copy(overlap_mean)
        overlap_spread[overlap_spread < zero] = 0
        overlap_spread[overlap_spread > 0] = 1
        ratio_eigvec_spread = np.sum(overlap_spread)/overlap_spread.shape[0]

        ## We start plotting procedures
        print('Plotting')
        # Set the path to figures
        if len(args.config) == 1:
            fig_path = FIGURE_DIR / dir / name
        else:
            fig_path = FIGURE_DIR / dir / 'Hessian_comparison_four_classes'
        fig_path.mkdir(parents=True, exist_ok=True)

        # Plot the Hessian spectrum
        if plot_spectra is True:
            ax_spectra[0].scatter(np.arange(len(heigenvalues))[-factor*num_classes:], heigenvalues[-factor*num_classes:], c = colors[model_no], label = name + ", Tr = " + str(np.sum(heigenvalues)) + ", 0 heigs = " + str(len(np.where(heigenvalues > zero)[0])) + ", model_l2 = " + str(model_l2_norm(model)))
            ax_spectra[1].scatter(np.arange(len(heigenvalues))[-factor*num_classes:], heigenvalues[-factor*num_classes:] / np.max(heigenvalues), c = colors[model_no], label = "scaled " + name + ", Tr = " + str(np.sum(heigenvalues  / np.max(heigenvalues))) + ", l_0 = " + str(len(np.where((heigenvalues  / np.max(heigenvalues)) > zero)[0])))
            ax_spectra[0].legend()
            ax_spectra[1].legend()

        # Plot a few largest eigenvectors for all setups
        if plot_eigenvectors is True:
            for eigenvector in range(how_many_largest_eigenvectors):
                for i in range(train_size):
                    axs_eigenvectors[eigenvector,model_no].plot(np.arange(len(heigenvectors[:, eigenvector])),heigenvectors[:, eigenvector], c=colors[model_no])
                    
            axs_eigenvectors[0,model_no].text(0.5, 1.1, name, horizontalalignment='center', verticalalignment='center', transform=axs_eigenvectors[0,model_no].transAxes, size=14)

        # Plot overlaps of all training gradients onto all Hessian eigenvectors for all setups
        if plot_overlaps is True:
            
            for i in range(train_size):
                axs_overlaps[0, model_no].plot(np.arange(len(heigenvectors[:, 0])), overlaps[:,i],
                                               c=colors[train_data.labels[i]])  # colors[model_no])
                axs_overlaps[0, model_no].set_ylim(-1,1)
                plt.grid(True)
            axs_overlaps[0,model_no].text(0.5, 1.1, name+' generalization measure: '+str(ratio_eigvec_spread), horizontalalignment='center', verticalalignment='center', transform=axs_overlaps[0,model_no].transAxes, size=14)

            # Plot overlaps of all training gradients onto largest Hessian eigenvectors for all setups
            for c in train_data.labels.unique():
                mask = train_data.labels == c
                axs_overlaps[1, model_no].plot(np.arange(len(heigenvectors[:, 0]))[-factor * num_classes:], overlaps[:,mask].mean(1)[-factor * num_classes:],
                                               c=colors[c], label = str(c),lw=3)
            for i in range(train_size):
                axs_overlaps[1, model_no].plot(np.arange(len(heigenvectors[:, 0]))[-factor * num_classes:],
                                               overlaps[:,i][-factor * num_classes:],
                                               c=colors[train_data.labels[i]],alpha=0.1)  # colors[model_no])

                axs_overlaps[1, model_no].set_ylim(-1,1)
                plt.grid(True)
        
        # Plot cosine similarity of all pairs of training loss gradients, ordered by classes, as heatmaps for all setups
        if plot_coherence is True:
            coherent_normalized_grads_matrix = np.zeros((train_size,train_size))
            grad_data = compute_grad(model, train_data, criterion, sort=True)
            for i in range(train_size):
                for j in range(train_size):
                    coherent_normalized_grads_matrix[i, j] = np.dot(grad_data[i]/np.linalg.norm(grad_data[i]),grad_data[j]/np.linalg.norm(grad_data[j]))
            for_normalized_colorbar = axs_normalized_coherence[0, model_no].imshow(coherent_normalized_grads_matrix, norm=SymLogNorm(linthresh=1e-6), cmap='RdBu')
            plt.colorbar(for_normalized_colorbar, ax = axs_normalized_coherence[0, model_no])
            axs_normalized_coherence[0,model_no].text(0.5, 1.1, name, horizontalalignment='center', verticalalignment='center', transform=axs_normalized_coherence[0,model_no].transAxes, size=14)
        
        # Plot cosine similarity of all pairs of training logit gradients, ordered by logit and classes, as heatmaps for all setups
        if plot_logit_coherence is True:
            coherent_normalized_logit_grads_matrix = np.zeros((num_classes*train_size,num_classes*train_size))
            grad_data = logit_gradients["normalized"]
            for i in range(num_classes*train_size):
                for j in range(num_classes*train_size):
                    coherent_normalized_logit_grads_matrix[i, j] = np.dot(grad_data[i]/np.linalg.norm(grad_data[i]),grad_data[j]/np.linalg.norm(grad_data[j]))
            idx = None
            for cls in range(3):
                zero_idx = np.zeros(50) + (cls*150)
                zero_idx = np.array([int(zero_idx[i]+(i)*3) for i in range(50)])
                one_idx = zero_idx + 1
                two_idx = zero_idx + 2
                if idx is None:
                    idx = zero_idx #np.concatenate((zero_idx, one_idx, two_idx), axis=0)
                else:
                    idx = np.concatenate((idx, zero_idx), axis=0) #np.concatenate((idx, zero_idx, one_idx, two_idx), axis=0)
            coherent_sorted = coherent_normalized_logit_grads_matrix[idx]
            coherent_sorted = coherent_sorted[:, idx]
            for_logit_normalized_colorbar = axs_logit_normalized_coherence[0, model_no].imshow(coherent_sorted, cmap='RdBu', vmin=-1, vmax=1) #norm=SymLogNorm(linthresh=1e-3), 
            plt.colorbar(for_logit_normalized_colorbar, ax = axs_logit_normalized_coherence[0, model_no])
            #plt.clim(-1, 1)
            #axs_logit_normalized_coherence[0, model_no].set_clim(-1, 1)
            axs_logit_normalized_coherence[0, model_no].text(0.5, 1.1, name, horizontalalignment='center', verticalalignment='center', transform=axs_logit_normalized_coherence[0,model_no].transAxes, size=14)
    
    ## Export figures
    if plot_spectra is True:
        fig_spectra.savefig(fig_path / f'Hessian_spectra{".sharp" if args.reparameterize else ""}.png')
    if plot_eigenvectors is True:
        fig_eigenvectors.savefig(fig_path / f'Hessian_eigenvectors{".sharp" if args.reparameterize else ""}.png')
    if plot_overlaps is True:
        fig_overlaps.savefig(fig_path / f'Hessian_overlaps{".sharp" if args.reparameterize else ""}.png')
    if plot_coherence is True:
        fig_normalized_coherence.savefig(fig_path / f'grads_normalized_coherence{".sharp" if args.reparameterize else ""}.png')
    if plot_logit_coherence is True:
        fig_logit_normalized_coherence.savefig(fig_path / f'grads_logit_normalized_coherence{".sharp" if args.reparameterize else ""}.png')
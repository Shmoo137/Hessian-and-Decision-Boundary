from settings import init, CMAP_OVERLAP, FIGWIDTH_COLUMN, FIGWIDTH_FULL

init()

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# plt.rcParams["font.family"] = "Times New Roman"


from src.config import *
from src.utils.plotting import *
from src.utils.general import find_hessian
from src.architectures import FNN, FNN_2layer, CNN
from src.utils.saving import dump_pickle, load_pickle
from src.hessian.generalization import compute_threshold, generalization_measure
from src.hessian.grads import compute_reinforcing_gradients, compute_grad_overlaps_w_heigenvectors
from src.datasets import GaussMixtureDataset, IrisDataset, HierachicalGaussMixtureDataset, data_to_torch, CircleDataset, HalfMoonDataset, GaussCheckerboardLinearClose, GaussCheckerboardNoisyClose, random_split

models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training', 'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'iris': IrisDataset, 'hierachical': HierachicalGaussMixtureDataset,
            'circle': CircleDataset, 'half_moon': HalfMoonDataset, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}



def generate_grid_data(min_x, max_x,min_y, max_y,dim_x, dim_y):
    x1 = np.linspace(min_x, max_x, dim_x)
    x2 = np.linspace(min_y, max_y, dim_y)
    X1,X2 = np.meshgrid(x1,x2)
    return X1,X2, np.vstack([X1.ravel(),X2.ravel()]).T


def show_training_points(ax,train_data):
    markers = ['o','d','+','x']
    label_classes = np.unique(train_data.labels)    
    for i, c in enumerate(label_classes):
        mask = train_data.labels == c
        ax.scatter(train_data.all[:,0][mask],train_data.all[:,1][mask], marker=markers[i],color='black', edgecolors='black',s=3) 

    # keep in case we want to add legend
    # legend_elements = [Line2D([0], [0], marker=marker, color='black', label=label, markersize=4) for label,marker in enumerate(markers[:label_classes])]

def show_decision_boundaries(ax,X1,X2,Y, as_scatter=False):
    img_sobel = sp.ndimage.sobel(Y)
    sbl_max = np.amax(abs(img_sobel))
    bn_img_direction_1 = np.abs(img_sobel) >= (sbl_max / 5.0)
    bn_img_direction_1 = bn_img_direction_1.reshape(X1.shape)

    img_sobel = sp.ndimage.sobel(Y, axis=0)
    sbl_max = np.amax(abs(img_sobel))
    bn_img_direction_2 = np.abs(img_sobel) >= (sbl_max / 5.0)
    bn_img_direction_2 = bn_img_direction_2.reshape(X1.shape)

    bn_img = bn_img_direction_1 + bn_img_direction_2

    if as_scatter:
        ax.scatter(X1[bn_img],X2[bn_img],c='black',s=3)
    else:
        ax.pcolor(X1,X2,bn_img,cmap='Greys')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute Hessians, spectra, eigenvectors, and overlaps and visualize them.')
    parser.add_argument('--config', type=str, nargs='+', help=f'Configuration file(s) of model(s) from {CONFIG_DIR}.')
    parser.add_argument('--overlap_vectors', help='Which type of vectors to compute overlaps with', default='top_heigenvectors') 
    parser.add_argument('--k', help='Number of eigenvectors to plot', default=3, type=int) 
    parser.add_argument('--hessian_subset', nargs='+',type=int,help='Which subset of class labels to use for the hessian. None is deafault and means using all data.', default=None)
    parser.add_argument('--resolution',type=int,help='How fine the grid should be in the x and y direction.', default=100)
    args = parser.parse_args()

    args.precomputed_hessian=False
    args.precomputed_overlap=False
    args.individual_colorbars=False
    args.reparameterize=False

    min_x = -2
    max_x = 2
    min_y = -4
    max_y = 4

    if args.hessian_subset is None: 
        args.hessian_subset = 'all'
    else:
        assert not args.precomputed_hessian, "Saving Hessian for subset of data is not supported."
    if args.reparameterize:
        assert not args.precomputed_hessian, "Saving Hessian for reparameterixation is not supported."
    
    n_configs=2

    num_rows=n_configs
    num_cols=args.k+1

    n_cols = num_cols
    n_rows = num_rows

    WIDTH_OVERLAP = 0.5 * FIGWIDTH_FULL
    ratio = np.ones(n_cols)
    ratio[-1] = 1.2
    fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(WIDTH_OVERLAP*0.95, WIDTH_OVERLAP/(n_cols+1)*n_rows),gridspec_kw={'width_ratios': ratio})

    ## Read config file and set model and data
    for model_no in range(n_configs):
        config_file = Path(args.config[model_no])
        ## Read config file and set model and data
        config_dir = Path('config')

        with open(CONFIG_DIR / config_file, 'r') as f:
            config = json.load(f)

        name = config_file.stem
        dir = config_file.parent


        figure_path = FIGURE_DIR / 'simplicity_bias'
        figure_path.mkdir(parents=True, exist_ok=True)

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

        if "loss" in config:
            if config['loss'] == 'mse':
                criterion = torch.nn.MSELoss()
            elif config['loss'] == 'hinge':
                criterion = torch.nn.MultiMarginLoss()
            elif config['loss'] == 'nll':
                criterion = torch.nn.NLLLoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))


        ## Calculate Hessian at the minimum along with its spectrum and eigenvectors
        # Run training data through the model
        if args.hessian_subset != 'all':
            for c in args.hessian_subset:
                assert c in train_data.labels, f"Class {c} not in training data."
            mask = train_data.labels == args.hessian_subset[0]
            for c in args.hessian_subset:
                mask = np.logical_or(mask,(train_data.labels == c))
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
            hessian = np.load(grad_path /  (name + '_hessian_identical.npy'))
            heigenvalues = np.load(grad_path /  (name + '_heigenvalues_identical.npy'))
            heigenvectors = np.load(grad_path /  (name + '_heigenvectors_identical.npy'))
        else:
            hessian = find_hessian(loss, model)
            print("Finding eigenvectors...")
            heigenvalues, heigenvectors = np.linalg.eigh(hessian)
            # if args.hessian_subset != 'all' or args.reparameterize:
            #     print('Sorry, not saving hessian of subset or for the reparameterization, this will mess up the saved Hessians so far.')
            # else:
            np.save(grad_path /  (name + '_hessian_identical'), hessian)
            np.save(grad_path /  (name + '_heigenvalues_identical'), heigenvalues)
            np.save(grad_path /  (name + '_heigenvectors_identical'), heigenvectors)
        

        # Select the vecwe are interested in...
        if args.overlap_vectors == 'top_heigenvectors':
            vectors = [heigenvectors[:,-which_heigenvector] for which_heigenvector in range(1,args.k+1)]
            weights = [heigenvalues[-which_heigenvector] for which_heigenvector in range(1,args.k+1)]
            heigs = [heigenvalues[-which_heigenvector] for which_heigenvector in range(1,args.k+1)]

        elif args.overlap_vectors == 'lowest_heigenvectors':
            vectors = [heigenvectors[:,which_heigenvector] for which_heigenvector in range(args.k)]
            weights = [heigenvalues[which_heigenvector] for which_heigenvector in range(args.k)]

        elif args.overlap_vectors == 'random_heigenvectors':
            rs = np.random.RandomState(42)
            heigs_choice= rs.randint(0,heigenvectors.shape[0],args.k)
            vectors = [heigenvectors[:,which_heigenvector] for which_heigenvector in heigs_choice]
            weights = [heigenvalues[which_heigenvector] for which_heigenvector in heigs_choice]

        elif args.overlap_vectors == 'random_directions':
            rs = np.random.RandomState(42)
            vectors = [rs.randn(heigenvectors.shape[0]) for _ in range(args.k)]
            weights = [1 for _ in range(args.k)]

        else:
            raise ValueError(f'Unknown overlap vector type {args.overlap_vectors}')

        if args.precomputed_overlap:
            print('Loading precomputed overlap')
            overlaps_data = load_pickle(grad_path / (name + '_overlaps_identical.pkl'))
        else:
            print('Computing overlap')
            vectors = [heigenvectors[:, which_heigenvector] for which_heigenvector in range(heigenvectors.shape[1])]
            overlaps_data = compute_reinforcing_gradients(model, train_data.all, vectors, criterion, mnist=True)
            dump_pickle(overlaps_data, grad_path / (name + '_overlaps_identical.pkl'))

        # print('Loading precomputed overlap')
        # overlaps_data = load_pickle(grad_path /  (name + '_overlaps.pkl'))

        # calc generalisation metric to display on plot
        eps_min, eps_avg = compute_threshold(model, train_data.all, rand_vec_dim=heigenvectors.shape[0], k=5)
        ratio_eigenvec_spread = generalization_measure(overlaps_data,eps_avg)

        # normalize the weighted vectors
        weights = np.array(weights)
        weights = weights / weights.sum()

        cmap_mesh = CMAP_OVERLAP

        X1,X2, X = generate_grid_data(min_x=min_x, max_x=max_x,min_y=min_y, max_y=max_y,dim_x=100, dim_y=100)
        X = data_to_torch(X.astype(np.float32))
        Y = np.argmax(model(X).detach().numpy(),axis=1)
        Y = Y.reshape(X1.shape)
        overlaps = compute_reinforcing_gradients(model, X, vectors, criterion)

        

        # Plot first element of the final plot, just the decision boundaries
        show_decision_boundaries(axs[model_no,0],X1,X2,Y)
        show_training_points(axs[model_no,0],train_data)
        if model_no==0: 
            axs[model_no,0].set_title(rf'decision boundary')
            axs[0,0].set_ylabel('normal\n'+r'$\mathcal{G}_{\theta}=$'+str(np.round(ratio_eigenvec_spread,5)))
        if model_no==1:
            axs[1,0].set_ylabel('wide margin\n'+r'$\mathcal{G}_{\theta}=$'+str(np.round(ratio_eigenvec_spread,5)))
        #axs[model_no,0].set_aspect(1)
        axs[model_no,0].xaxis.set_tick_params(labelsize=9)
        axs[model_no,0].yaxis.set_tick_params(labelsize=9)

        fig.tight_layout(h_pad=0.5)
        # Plot the rest of the final plot, the decision boundaries and the overlap
        for i in range(len(vectors)):
            overlap = overlaps[i,:].reshape(X1.shape)
            im = axs[model_no,i+1].pcolormesh(X1,X2,overlap,cmap=cmap_mesh,shading='nearest')
            axs[model_no,i+1].set_xticks([])
            axs[model_no,i+1].set_yticks([])
            axs[model_no,i+1].set_xlabel(rf'$\lambda_{i+1}=$'+str(np.round(heigs[i],4)), fontsize=11)
            #axs[model_no,i+1].set_aspect(1)
            if args.individual_colorbars:
                fig.colorbar(im, ax=axs[model_no,i+1])
            else:
                im.set_clim(-1.,1.)
            # show_decision_boundaries(axs[model_no,i+1],X1,X2,Y,as_scatter=True)
            show_training_points(axs[model_no,i+1],train_data)
            if model_no==0: axs[model_no,i+1].set_title(rf'$v_{i+1}$')

        if not args.individual_colorbars:
            fig.colorbar(im, ax=axs[model_no,-1], label='Alignment')

        # fig.suptitle(f'{dir}: [{name}] {args.k} {args.overlap_vectors}')
        
        hs = "".join([str(i) for i in args.hessian_subset])
    plt.savefig(figure_path / f'simplicity_bias_{name}_{args.overlap_vectors}_{args.k}_normcolor={not args.individual_colorbars}_hessian_subset={hs}_reparameterized={args.reparameterize}.overlap.png', bbox_inches='tight', dpi = 600)
    plt.close()


from settings import init, CMAP_OVERLAP, FIGWIDTH_COLUMN

init()

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import combinations


from src.config import *
from src.utils.plotting import *
from src.utils.saving import dump_pickle, load_pickle
from src.architectures import FNN, FNN_2layer, CNN
from src.utils.general import find_hessian
from src.hessian.grads import compute_reinforcing_gradients
from src.hessian.grads import design_grad_parallel_to_heigenvector, compute_grad_overlaps_w_heigenvectors
from src.datasets import GaussMixtureDataset, IrisDataset, data_to_torch
from src.datasets import GaussMixtureDataset, IrisDataset, MNIST2DDataset, GaussCheckerboardLinearClose
from src.datasets import random_split

models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN]}
setups = {'normal': 'normal_training', 'random': 'random_label_training', 'adv': 'adversarial_init_training'}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
datasets = {'gauss_mixtures': GaussMixtureDataset, 
            'iris': IrisDataset, 'mnist2D': MNIST2DDataset, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}

artificial_maxoverlappoint = True

def show_training_points(train_data):
    markers = ['s','o','d','+','x']
    label_classes = np.unique(train_data.labels)    
    for i, c in enumerate(label_classes):
        mask = train_data.labels == c
        plt.scatter(train_data.all[:,0][mask],train_data.all[:,1][mask], marker=markers[i],color='black', edgecolors='black',s=3) 


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
    args = parser.parse_args()
    args.precomputed_hessian=True
    args.precomputed_overlap=True

    ## Read config file and set model and data
    config_file = Path(args.config[0])
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

    train_size =  int(config['train_fraction'] * n)
    test_size = int(config['test_fraction'] * n)
    val_size = n - train_size - test_size

    train_data, test_data, val_data = random_split(dataset, [train_size,test_size,val_size], generator=torch.Generator().manual_seed(42))

    model = model_cls(input_size=input_size, num_classes=num_classes, **config['model']['args'])

    criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(MODEL_DIR / dir / (name + '.pt')))

    outputs = model(train_data.all)
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
        heigenvalues, heigenvectors = np.linalg.eigh(hessian)
        np.save(grad_path /  (name + '_hessian'), hessian)
        np.save(grad_path /  (name + '_heigenvalues'), heigenvalues)
        np.save(grad_path /  (name + '_heigenvectors'), heigenvectors)
    

    ## What is encoded by the Hessian eigenvectors corresponding to the largest eigenvalues?
    # Check its overlaps with per-example gradients
    if args.precomputed_overlap:
        print('Loading precomputed overlap')
        overlaps_dict = load_pickle(grad_path /  (name + '_overlaps_identical.pkl'))
    else:
        print('Computing overlap')
        vectors = [heigenvectors[:, which_heigenvector] for which_heigenvector in range(heigenvectors.shape[1])]
        overlaps = compute_reinforcing_gradients(model, train_data.all, vectors, criterion)
        dump_pickle(overlaps,grad_path /  (name + '_overlaps.pkl'))

    # Plot the overlaps
    plt.rcParams["figure.figsize"] = (20, 5)
    possible_combinations = list(combinations(np.arange(input_size), 2))
    num_combinations = len(possible_combinations)
    
    ## We start plotting procedures
    print('Plotting')
    # Set the path to figures
    fig_path = FIGURE_DIR / dir / name
    fig_path.mkdir(parents=True, exist_ok=True)
    (fig_path  / 'grads_vs_heigenvectors').mkdir(parents=True, exist_ok=True)

    fig_path = FIGURE_DIR / dir
    fig_path.mkdir(parents=True, exist_ok=True)
    (fig_path  / 'Hessian_comparison').mkdir(parents=True, exist_ok=True)


    ## Find an artificial data point that has a maximal overlap with a selected heigenvector
    which_heigenvector = heigenvectors.shape[0] - 1
    # choose largest heigenvector
    perfect_representative = design_grad_parallel_to_heigenvector(which_heigenvector, model, heigenvectors, input_size=input_size, num_classes=num_classes)
    representative_coord=np.array([perfect_representative[0].item(), perfect_representative[1].item()])
    
    ## Find the training point w largest overlap w top heigenvector
    # heigenvectors[:, which_heigenvector]
    # index with max overlap 
    k=1
    vectors = [heigenvectors[:,-which_heigenvector] for which_heigenvector in range(1,k+1)]
    X_data = train_data.all.numpy()
    X1 = data_to_torch(X_data.astype(np.float32))
    overlaps = compute_reinforcing_gradients(model, X1, vectors, criterion)
    
    training_set_size = len(train_data)

    #naive,to optimise
    oldmax=-1e7
    for i in range(training_set_size):
        maximum2=np.max(overlaps[0,i])
        if oldmax<maximum2:
            ind2=int(i)
            oldmax=maximum2
    oldmax=1e7
    for i in range(training_set_size):
        maximum3=np.min(overlaps[0,i])
        if oldmax>maximum3:
            ind3=int(i)
            oldmax=maximum3

    min_x = -2
    max_x = 2
    min_y, max_y = -4,4
    dim_x = dim_y = 400
    x = np.linspace(min_x, max_x, dim_x)
    y = np.linspace(min_y, max_y, dim_y)

    a2=X_data[ind2,:]
    l2norm2 = np.linalg.norm(a2-representative_coord)
    a3=X_data[ind3,:]
    l2norm3 = np.linalg.norm(a3-representative_coord)
    minl2=np.min([l2norm2, l2norm3])


    X,Y = np.meshgrid(x,y)

    def z_function(x,y):
        r = model(data_to_torch(np.array([[x,y]]))).detach().numpy()#math.sqrt(x**2 + y**2)
        return np.argmax(r)

    z = np.array([z_function(x,y) for (x,y) in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    latexify(fig_width=4, fig_height=4)


    fig, ax = plt.subplots()

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    # plt.pcolormesh(X,Y,Z,cmap=cmap,shading='nearest')
    show_decision_boundaries(ax,X,Y,Z)
    ax.scatter(X_data[ind2,0], X_data[ind2,1], s=100, c="purple", alpha=0.5)
    ax.scatter(X_data[ind2,0], X_data[ind2,1], s=100, c="black", marker="+", label=r'$x_t^{max}$')
    ax.scatter(X_data[ind3,0], X_data[ind3,1], s=100, c="orange", alpha=0.5)
    ax.scatter(X_data[ind3,0], X_data[ind3,1], s=100, c="black", marker="_", label=r'$x_t^{min}$')
    ax.scatter(perfect_representative[0].item(), perfect_representative[1].item(), s=100, c="red",label=r'$x_b$')
    show_training_points(train_data)
    ax.set_title(r'$L^2_{max} \wedge L^2_{min}=$ '+str(np.round(minl2,3)),fontsize=20)
    # ax.legend(fontsize=12)
    ax.set_xticks([-2,0,2])
    ax.set_yticks([-4,-2,0,2,4])
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    figure_path = FIGURE_DIR / 'margin_estimate'
    plt.savefig(figure_path / f'{name}_margin_estimation_model2_1_new.png', bbox_inches='tight', dpi = 600)
    plt.close()

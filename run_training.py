import argparse
import json
from pathlib import Path

import torch

from src.architectures import *
from src.config import *
from src.datasets import *
from src.training import train_and_validate
from src.utils.plotting import *


models = {cls.name: cls for cls in [FNN, FNN_2layer, CNN, LeNet5, ResNet18]}
optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
schedulers = {'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR}
datasets = {'gauss_mixtures': GaussMixtureDataset, 'iris': IrisDataset,
            'mnist2D': MNIST2DDataset, 'cifar10': CIFAR10, 'hierachical': HierachicalGaussMixtureDataset,
            'circle': CircleDataset, 'half_moon': HalfMoonDataset, 'intro': IntroDataset,
            'gauss_checkerboard_noisy_close': GaussCheckerboardNoisyClose, 'gauss_checkerboard_linear_close': GaussCheckerboardLinearClose}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training.')

    parser.add_argument('--config', type=str,
                        help=f'Configuration file from {CONFIG_DIR}.')
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
    print('train test val total size ', train_size, test_size, val_size, n)
    if test_size == 0.0:
        train_data = test_data = val_data = MySubset(dataset, range(n))
    else:
        train_data, test_data, val_data = random_split(
            dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    s = torch.random.seed()
    print('random seed ', s)
    model = model_cls(input_size=input_size,
                      num_classes=num_classes, **config['model']['args'])

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

    # Don't use weight decay, momentum and scheduler for adversarial initialization!
    optimizer = optimizer_cls(model.parameters(), **
                              config['optimizer']['args'])
    if config['scheduler'] is False:
        scheduler = False
    else:
        scheduler_cls = schedulers[config['scheduler']['type']]
        scheduler = scheduler_cls(optimizer, **config['scheduler']['args'])

    import copy

    if config['init_model_at'] != "":
        if 'large_norm' in config:
            if config['large_norm'] == True:
                # Scale up the norm of the model according to the one with the adversarial init
                with torch.no_grad():
                    init = copy.deepcopy(model.state_dict())
                    total_norm_init = model_l2_norm(model)
                    print('Original norm : ', total_norm_init)

                    model.load_state_dict(torch.load(
                        MODEL_DIR / (config['init_model_at'] + '.pt')))
                    total_norm_other = model_l2_norm(model)
                    print('Scaled up norm ', total_norm_other)

                    model.load_state_dict(init)
                    parameters = model.parameters()
                    for p in parameters:
                        p.data = p.data * total_norm_other / total_norm_init
                    total_norm_scaled = torch.norm(torch.stack(
                        [torch.norm(g.detach(), 2.0) for g in model.parameters()]), 2.0)
            else:
                # directly use the adversarial init
                model.load_state_dict(torch.load(
                    MODEL_DIR / (config['init_model_at'] + '.pt')))

        else:
            # directly use the adversarial init
            model.load_state_dict(torch.load(
                MODEL_DIR / (config['init_model_at'] + '.pt')))

    final_losses = train_and_validate(model, train_data, val_data, test_data, optimizer,
                                      criterion=criterion, scheduler=scheduler,
                                      name=name, folder_model=MODEL_DIR / dir,
                                      folder_grads=GRAD_DIR / dir,
                                      early_stopping=config.get(
                                          'early_stopping', False),
                                      neural_collapse=False, plot=False,
                                      **config['trainer'])

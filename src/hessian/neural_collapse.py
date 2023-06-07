import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm  # for the progress bar outside Jupyter
#from tqdm.notebook import tqdm #  for the progress bar in Jupyter notebook

from src.datasets import data_to_torch
from src.utils.general import get_lr

# Compute the within-class mean and variance
# Now for the final output layer
# TODO: change to the one-before-last layer with the feature hook!
def compute_neural_collapse(model, data, num_classes):
  means_dict = {}
  var_dict = {}

  for i in range(num_classes):
    which_class = "class" + str(i)
    predictions = model(data[which_class])
    means_dict[which_class] = np.mean(predictions.detach().numpy(), axis = 0)
    var_dict[which_class] = np.std(predictions.detach().numpy(), axis = 0)

  return means_dict, var_dict

# "Hook" placed on a certain layer with name "module" allows to save its activations given the input to features_blobs list
def hook_feature(module, input, output):
  features_blobs = []
  features_blobs.append(output.data.cpu().numpy())
  return features_blobs

# Find an input vector whose gradient is parallel to the chosen heigenvector
# TODO: change to the one-before-last layer with the feature hook!
def design_input_parallel_to_NC_class(mean, model, input_size = 4, input_criterion = nn.MSELoss(), 
                                        num_epochs = 20000, learning_rate = 0.01, weight_decay = 0.001,
                                        milestones = [5000,10000,12000,15000,17000], plot = True):

    # Initialize the input vector to be optimized
    input = data_to_torch(np.zeros(input_size))
    torch.nn.init.normal_(input, mean = 0.0, std = 1.0)
    input.requires_grad = True

    optimizer = torch.optim.SGD([input], lr = learning_rate, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.1)

    # Optimize the input vector
    hold_loss = []

    for epoch in tqdm(range(num_epochs), desc='Training'):

        # To compute the gradient of the loss of the input vector
        outputs = model(input)#.reshape(-1, num_classes)
        loss = input_criterion(outputs, torch.Tensor(mean))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hold_loss.append(loss.item())
        
        if epoch % 500 == 0:
            print("Epoch: ", epoch, "/", num_epochs, "Loss: ", loss.item(), "LR: ", get_lr(optimizer))
        
        if scheduler is not False:
            scheduler.step() # here for milestones scheduler    

        if plot is True:
            # Plot the losses
            plt.figure()
            plt.plot(hold_loss, label="training loss")
            plt.xlabel('epochs')
            plt.title("Losses plots")
            plt.ylim(1e-10,1)
            plt.yscale("log")
            plt.legend()
            plt.show()
            plt.close()
        
        return input

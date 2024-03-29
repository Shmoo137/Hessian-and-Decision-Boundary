import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm # for the progress bar

from src.datasets import data_to_torch
from src.utils.general import flatten_grad, get_lr

# Compute gradients of single data points of a model according to the criterion and labels
def compute_single_grads(model, data, labels, criterion = nn.CrossEntropyLoss()):
  grads_dict = {}

  for i, input_data in enumerate(data):
    output_for_grads = model(input_data)

    loss_for_grads = criterion(output_for_grads, labels[i])
    gradient = torch.autograd.grad(loss_for_grads, model.parameters(), create_graph = False)
    gradient = flatten_grad(gradient)
    normalized_gradient = gradient / torch.sqrt(torch.sum(gradient**2))

    grads_dict[i] = {"grad": gradient, "normalized_grad": normalized_gradient}
  
  return grads_dict

# Find an input vector whose gradient is parallel to the chosen heigenvector
def design_grad_parallel_to_heigenvector(which_heigenvector, model, heigenvectors, input_size = 4, num_classes = 3,
                                        model_criterion = nn.CrossEntropyLoss(), input_criterion = nn.MSELoss(), 
                                        num_epochs = 20000, learning_rate = 0.01, weight_decay = 0.001,
                                        milestones = [5000], plot = True):
  
    # Initialize the input vector to be optimized
    input = data_to_torch(np.zeros(input_size))
    torch.nn.init.normal_(input, mean = 0.0, std = 1.0)
    input.requires_grad = True

    optimizer = torch.optim.SGD([input], lr = learning_rate, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.1)

    # Optimize the input vector
    hold_loss = []

    for epoch in tqdm(range(num_epochs), desc='Finding optimal parallel grad'):

        # To compute the gradient of the loss of the input vector
        outputs = model(input).reshape(-1, num_classes)
        _, predicted = torch.max(outputs.data, 1)

        loss = model_criterion(outputs, predicted)
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad = flatten_grad(grad)
        norm = torch.sum(grad**2)

        # Forward pass
        loss_to_minimize = input_criterion(
                                torch.dot(grad, torch.Tensor(heigenvectors[:, which_heigenvector]))**2 / norm, 
                                torch.tensor(1).float())

        # Backward and optimize
        optimizer.zero_grad()
        loss_to_minimize.backward()
        optimizer.step()

        hold_loss.append(loss_to_minimize.item())
        
        if epoch % 500 == 0:
            print("Epoch: ", epoch, "/", num_epochs, "Loss: ", loss_to_minimize.item(), "LR: ", get_lr(optimizer))
        
        if scheduler is not False:
            scheduler.step() # here for milestones scheduler    

    # if plot is True:
    #     # Plot the losses
    #     plt.figure()
    #     plt.plot(hold_loss, label="training loss")
    #     plt.xlabel('epochs')
    #     plt.title("Losses plots")
    #     plt.legend()
    #     plt.show()
    #     plt.close()
    
    return input

def compute_grad(model, train_data, criterion = nn.CrossEntropyLoss(), sort=False):
    grad_data = []
    if sort:
        sorting_array = np.argsort(train_data.labels.flatten())
    else:
        sorting_array = np.arange(len(train_data.labels))

    if "CNN" in model.__class__.__name__:
        data = torch.unsqueeze(train_data.all, 1)
        labels = torch.unsqueeze(train_data.labels, 1)
    else:
        data = train_data.all
        labels = train_data.labels

    for i in tqdm(np.arange(len(train_data.labels))):
        output = model(data[sorting_array[i]])
        y_argmax = np.argmax(output.detach())
        if criterion._get_name() == 'MSELoss':
            loss = criterion(output, nn.functional.one_hot(y_argmax, num_classes=output.shape[1]).to(torch.float32))
        elif criterion._get_name() == 'NLLLoss':
            loss = criterion(output.reshape(1, -1), y_argmax.reshape(1))
        else:
            loss = criterion(output.reshape(1, -1), y_argmax.reshape(1))
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        grad = flatten_grad(grad).reshape(1,-1)
        #grad_data.append(grad)
        try:
            grad_data = torch.cat((grad_data, grad), dim=0)
        except:
            grad_data = grad
    return grad_data


# Compute overlaps of per-example gradients with heigenvectors:
def compute_grad_overlaps_w_heigenvectors(model, train_data, heigenvectors, criterion = nn.CrossEntropyLoss()):
  overlaps_dict = {}

  sorting_array = np.argsort(train_data.labels.flatten())

  if "CNN" in model.__class__.__name__:
    data = torch.unsqueeze(train_data.all, 1)
    labels = torch.unsqueeze(train_data.labels, 1)
  else:
    data = train_data.all
    labels = train_data.labels

  for i in tqdm(np.arange(len(train_data.labels)), desc = "Computing gradients and their projections onto heigenvectors"):
    #print('train_data.all[i]',train_data.all[i])
    output = model(data[sorting_array[i]])
    #print('output', output)
    #print('train_data.labels[i]',train_data.labels[i])
    loss = criterion(output, labels[sorting_array[i]])
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad = flatten_grad(grad)
    norm = torch.sum(grad**2)

    overlapping_array = []
    for j in range(len(heigenvectors[:,0])):
      overlap = np.dot(grad, heigenvectors[:,j])#**2
      sign = np.sign(overlap)
      overlap = sign * overlap**2 / norm
      overlapping_array.append(overlap)
    overlapping_array = np.array(overlapping_array)

    overlaps_dict[i] = {"overlaps": overlapping_array, "label": train_data.labels[i], "grad": grad.cpu().detach().numpy(), "grad_norm": norm}

  return overlaps_dict


# compute overlap of some data point with a specific heigenvector
def compute_overlap_of_point_w_heigenvector(model, x, y, heigenvectors, which_heigenvector, criterion = nn.CrossEntropyLoss(), two_dims = True, means = torch.Tensor([]), which_2_features = np.array([])):
  overlap_matrix=np.zeros((len(x),len(y)))
  overlap_matrix_normed_sign=np.zeros((len(x),len(y)))

  for i in tqdm(range(len(x)), desc = "Computing meshgrid overlaps w top heig"):
    for j in range(len(y)):
      if two_dims is True:
        output = model(torch.Tensor([x[i],y[j]]))     # loop through all points in grid
      else:
        helpful_index = 0
        input_size = len(means) + 2
        input_tensor = torch.zeros(input_size)
        for element in range(input_size):
          if element == which_2_features[0]:
            input_tensor[element] = x[i]
          elif element == which_2_features[1]:
            input_tensor[element] = y[j]
          else:
            input_tensor[element] = means[helpful_index]
            helpful_index = helpful_index + 1
          output = model(input_tensor)

      label_ij=np.argmax(output.detach().numpy())   # take output of x,y run through the model
      x1 = torch.tensor(label_ij)                   # tensor output of x,y in the grid
      loss = criterion(torch.Tensor(output), x1)    # compute loss
      grad = torch.autograd.grad(loss, model.parameters(), create_graph=False) # get gradient
      grad = flatten_grad(grad)
      norm = torch.sum(grad**2)                     # get norm of gradient
      # for each datapoint compute its overlap with the top heigenvector indexed as 'which_heigenvector'
      overlap = np.dot(grad, heigenvectors[:,which_heigenvector])
      overlap_matrix[i,j]=overlap
      sign = np.sign(overlap)
      overlap = sign * overlap**2 / norm
      overlap_matrix_normed_sign[i,j]=overlap
  return overlap_matrix, overlap_matrix_normed_sign # matrices are len(x) x len(y) - need to transpose!

def analysis_overlap_w_vector(grad, vectors, cos_sim=True):
  """
  Compute the overlap of a gradient with a set of vectors.
  """
  overlaps = np.dot(vectors, grad)
  if np.linalg.norm(grad) > 1e-20 and cos_sim:
      overlaps = np.dot(vectors, grad) / (np.linalg.norm(grad)) #*np.linalg.norm(vectors, axis=1))
      """if np.any(overlaps > 1.1) or np.any(overlaps < -1.5):
         print(np.dot(vectors, grad))
         print(np.linalg.norm(grad))
         exit() """
  else:
     overlaps = np.zeros(len(vectors))
  return overlaps


def compute_reinforcing_gradients(model, X, vectors, criterion = nn.CrossEntropyLoss(), mnist=False, cos_sim=True, y=None, grads_label='reinforcing'):
  analysis_func=lambda g: analysis_overlap_w_vector(g,vectors,cos_sim)
  overlap_matrix=np.zeros((len(vectors),X.shape[0]))


  for i in tqdm(range(X.shape[0])):
    if not mnist:
        x = X[i].reshape(1, -1)
    else:
        if len(X[i].shape) == 3:
            x = X[i].reshape(1, X[i].shape[0], X[i].shape[1], X[i].shape[2])
        else:
            x = X[i]
    output = model(x)

    if grads_label == 'reinforcing':
      label = np.argmax(output.detach())
    elif grads_label == 'random':
      label = torch.randint(0, 3, (1,)).long()
    elif grads_label == 'zero':
      label = torch.Tensor([0]).long()
    elif grads_label == 'one':
      label = torch.Tensor([1]).long()
    elif grads_label == 'two':
      label = torch.Tensor([2]).long()
    else:
      print("Not implemented type of gradient labels, computing reinforcing gradients instead!")
      label = np.argmax(output.detach())

    if criterion._get_name() == 'MSELoss':
        loss = criterion(output, nn.functional.one_hot(label, num_classes=output.shape[1]).to(torch.float32))
    elif criterion._get_name() == 'NLLLoss':
        loss = criterion(output.reshape(1, -1), label.reshape(1))
    else:
        loss = criterion(output.reshape(1,-1), label.reshape(1))
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad = flatten_grad(grad)

    overlap_matrix[:,i] = analysis_func(grad)


  return overlap_matrix

def compute_grad_norm(model, X, criterion = nn.CrossEntropyLoss()):
  overlap_matrix=np.zeros(X.shape[0])


  for i in tqdm(range(X.shape[0])):
    output = model(X[i])
    y_argmax=np.argmax(output.detach())
    loss = criterion(output, y_argmax)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad = flatten_grad(grad)

    overlap_matrix[i] = np.dot(grad, grad)


  return overlap_matrix

def compute_grads(model, X, criterion = nn.CrossEntropyLoss()):
  overlap_matrix=[]


  for i in tqdm(range(X.shape[0])):
    output = model(X[i])
    y_argmax=np.argmax(output.detach())
    loss = criterion(output, y_argmax)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad = flatten_grad(grad).detach().numpy() 

    overlap_matrix.append(grad / np.linalg.norm(grad))

  return np.array(overlap_matrix)

def compute_logit_grads(model, train_data, mnist=False, sort=False):
  if sort:
    sorting_array = np.argsort(train_data.labels.flatten())
  else:
    sorting_array = np.arange(len(train_data.labels))
  
  if mnist:
    data = torch.unsqueeze(train_data.all, 1)
    labels = torch.unsqueeze(train_data.labels, 1)
  else:
    data = train_data.all
    labels = train_data.labels

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  logit_grads_dict = {}
  logit_grads_matrix = np.array([])
  normalized_logit_grads_matrix = np.array([])

  for i in tqdm(np.arange(len(labels))):
    output = model(data[sorting_array[i]])
    for j in range(output.shape[0]):
      logit_grad_per_class = torch.autograd.grad(output[j], model.parameters(), create_graph=True) # we need create_graph = True because we're going for every element of output separately and having c passes
      logit_grad_per_class = flatten_grad(logit_grad_per_class).detach().numpy() 
      normalized_logit_grad_per_class = logit_grad_per_class / np.linalg.norm(logit_grad_per_class)

      logit_grads_matrix = np.append(logit_grads_matrix, logit_grad_per_class)
      normalized_logit_grads_matrix = np.append(normalized_logit_grads_matrix, normalized_logit_grad_per_class)

  logit_grads_dict = {"unnormalized": logit_grads_matrix.reshape(-1, num_params), "normalized": normalized_logit_grads_matrix.reshape(-1, num_params)}
  
  return logit_grads_dict
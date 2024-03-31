import torch
import numpy as np
import torch.nn as nn
import copy

from tqdm import tqdm  # for the progress bar outside Jupyter
#from tqdm.notebook import tqdm #  for the progress bar in Jupyter notebook

from matplotlib import pyplot as plt
import pickle

from src.hessian.grads import compute_single_grads
from src.hessian.neural_collapse import compute_neural_collapse, hook_feature



# Function to end all functions: training, gradients during the training, and neural collapse
def train_and_validate(model, train_data, validation_data, test_data,
                       optimizer, batch_size, num_epochs=2000, criterion=nn.MSELoss(), scheduler=False,
                       how_often_save_model=0, name="experiment", folder_model="./results/models/",
                       how_often_compute_grads=0, folder_grads="./results/grads/",
                       neural_collapse=False, plot=False, how_often_print=500, early_stopping=False):

    if batch_size == 'full':
        batch_size = len(train_data)


    folder_grads.mkdir(parents=True, exist_ok=True)
    folder_model.mkdir(parents=True, exist_ok=True)

    hold_loss = []
    hold_val_loss = []
    hold_test_loss = []
    stop = False

    models_dict = {}
    grads_dict = {}

    if neural_collapse == True:
        model._modules.get("0").register_forward_hook(hook_feature)
        means_dict = {}
        var_dict = {}
        num_classes = len(torch.unique(test_data["labels"]))

    train_iterator = torch.utils.data.DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=batch_size)

    if how_often_save_model != 0:
        models_dict[-1] = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(num_epochs), desc='Training'):

        for phase in ['train', 'val']:

            if phase == 'train':

                epoch_loss = 0.0

                for (X, y) in train_iterator:

                    # Forward pass
                    outputs = model(X.float())
                    if criterion._get_name() == 'MSELoss':
                        loss = criterion(outputs, nn.functional.one_hot(y).to(torch.float32))
                    elif criterion._get_name() == 'HingeEmbeddingLoss':
                        loss = criterion(outputs, y.reshape(-1,1))
                    elif criterion._get_name() == 'NLLLoss':
                        loss = criterion(outputs, y)
                    else:
                        loss = criterion(outputs, y)
            
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
            

                hold_loss.append(epoch_loss)


                if how_often_compute_grads != 0:
                    if epoch % how_often_compute_grads == 0:
                        grads_dict[epoch] = compute_single_grads(model, train_data.all, train_data.labels,
                                                                 criterion=criterion)

                if neural_collapse == True:
                    # Checking neural collapse
                    means_dict[epoch], var_dict[epoch] = compute_neural_collapse(model, train_data, num_classes)

            if phase == 'val':
                # model.eval()  # Set model to evaluating mode

                with torch.no_grad():
                    validation_itr = torch.utils.data.DataLoader(validation_data,
                                                                shuffle=False,
                                                                batch_size=validation_data.__len__())
                    val_dataset, val_labels = next(iter(validation_itr))
                    outputs = model(val_dataset.float())
                    if criterion._get_name() == 'MSELoss':
                        val_loss = criterion(outputs, nn.functional.one_hot(val_labels))
                    elif criterion._get_name() == 'HingeEmbeddingLoss':
                        val_loss = criterion(outputs, val_labels.reshape(-1, 1))
                    elif criterion._get_name() == 'NLLLoss':
                        val_loss = criterion(outputs, val_labels)
                    else:
                        val_loss = criterion(outputs, val_labels)

                    test_itr = torch.utils.data.DataLoader(test_data,
                                                            shuffle=False,
                                                            batch_size=test_data.__len__())
                    test_dataset, test_labels = next(iter(test_itr))
                    outputs = model(test_dataset.float())
                    if criterion._get_name() == 'MSELoss':
                        test_loss = criterion(outputs, nn.functional.one_hot(test_labels))
                    elif criterion._get_name() == 'HingeEmbeddingLoss':
                        test_loss = criterion(outputs, test_labels.reshape(-1, 1))
                    elif criterion._get_name() == 'NLLLoss':
                        test_loss = criterion(outputs, test_labels)
                    else:
                        test_loss = criterion(outputs, test_labels)

                hold_val_loss.append(val_loss.item())
                hold_test_loss.append(test_loss.item())

                if how_often_save_model != 0:
                    if epoch % how_often_save_model == 0:
                        models_dict[epoch] = copy.deepcopy(model.state_dict())

                if epoch % how_often_print == 0:
                    train_acc = accuracy(model, train_data.all, train_data.labels)
                    test_acc = accuracy(model, test_data.all, test_data.labels)
                    print("Epoch: ", epoch, "/", num_epochs, "Training loss: {:.4e}".format(loss.item()),
                          "Validation loss: {:.4e}".format(val_loss.item()),
                          "/", "Train acc: {:.1f} ".format(train_acc), "Test acc: {:.1f}".format(test_acc))

                    if early_stopping and train_acc == 100:
                        stop = True

                if scheduler is not False:
                    scheduler.step()

        if stop:
            print("Stopping training due to 100% training accuracy.")
            break

    # Save the model checkpoint
    torch.save(model.state_dict(), folder_model / ( name + '.pt'))
    # torch.save(list(model.parameters()), 'parameters_' + str(training_set_size) + '.pt')
    
    # Save results
    if how_often_save_model != 0:
        models_dict[epoch] = copy.deepcopy(model.state_dict())
        with open(folder_model / ("models_dict" + name + ".pkl"), "wb") as tf:
            pickle.dump(models_dict, tf)

    if how_often_compute_grads != 0:
        grads_dict[epoch] = compute_single_grads(model, train_data["all"], train_data["labels"], criterion=criterion)
        with open(folder_grads / ("grads_dict" + name + ".pkl"), "wb") as tf:
            pickle.dump(grads_dict, tf)

    if plot == True:
        # Plot the losses
        plt.figure()
        plt.plot(hold_loss, label="training loss")
        plt.plot(hold_val_loss, label="validation loss")
        plt.xlabel('epochs')
        plt.title("Losses plots")
        # plt.ylim(0,1)
        plt.legend()
        plt.show()
        plt.close()

        if neural_collapse == True:
            # Plot the neural collapse
            layer_size = len(means_dict[epoch]["class0"])
            plt.figure()
            for c in range(num_classes):
                for j in range(layer_size):
                    plt.errorbar(np.arange(num_epochs),
                                 np.array([means_dict[i]["class" + str(c)][j] for i in range(num_epochs)]),
                                 yerr=np.array([var_dict[i]["class" + str(c)][j] for i in range(num_epochs)]),
                                 label="class" + str(c) + ", output" + str(j))
            plt.xlabel('epochs')
            plt.title("Neural collapse")
            plt.legend()
            plt.show()
            plt.close()

    return np.array([hold_loss[-1], hold_val_loss[-1], hold_test_loss[-1]])


def accuracy(model, inputs, targets):
    preds = model(inputs.float()).argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy().astype(np.float32)
    return 100 * sum(preds == targets) / len(targets)

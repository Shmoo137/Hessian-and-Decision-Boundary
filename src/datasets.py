from random import seed
import numpy as np
from sklearn.datasets import make_circles, make_moons
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import _accumulate,randperm, T_co, Optional, Generator,Sequence, default_generator, List, Subset, T
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from src.config import DATASET_DIR

def fun(arg):
  return np.sin(arg[0]) + np.cos(arg[1]) * arg[2] + arg[3] * arg[4]

def dataToTorch(array):
  return Variable(torch.from_numpy(array).float())

def data_to_torch(data):
  data = Variable(torch.from_numpy(data))
  return data

def labelToTorch(array):
  return Variable(torch.from_numpy(array).float())

def labels_to_torch(labels):
  labels = Variable(torch.from_numpy(labels)).long()
  return labels

def generate_data(size = 3, input_size = 5, value_range = 5, noise = 0):
  data_array = np.array([])
  label_array = np.array([])

  for i in range(size):
    new_data = np.random.random(input_size) * value_range
    data_array = np.append(data_array, new_data)
    y = fun(new_data)
    label_array = np.append(label_array, y + np.random.uniform(-1.0, 1.0) * y * noise)

  data_array = data_array.reshape(size, -1)
  label_array = label_array.reshape(size, -1)

  return (dataToTorch(data_array), labelToTorch(label_array))

def separate_into_3_classes(data, labels):
  X_0 = np.array([])
  X_1 = np.array([])
  X_2 = np.array([])

  no_features = np.round((data.size/len(labels))).astype(int)

  for i, label in enumerate(labels):
    if label == 0:
      X_0 = np.append(X_0, data[i])
    elif label == 1:
      X_1 = np.append(X_1, data[i])
    elif label == 2:
      X_2 = np.append(X_2, data[i])

  return np.reshape(X_0, (-1, no_features)), np.reshape(X_1, (-1, no_features)), np.reshape(X_2, (-1, no_features))

def fit_standarize(data):
  mean = np.mean(data, axis = 0)
  std = np.std(data, axis = 0)
  data_to_scale = (data - mean) / std
  return data_to_scale, mean, std
  
def fit_normalize(data):
  min = np.amin(data, axis = 0)
  max = np.amax(data, axis = 0)
  data_to_scale = (data - min) / (max - min)
  return data_to_scale, min, max

def standarize(data, mean, std):
  data_to_scale = (data - mean) / std
  return data_to_scale

def normalize(data, min, max):
  data_to_scale = (data - min) / (max - min)
  return data_to_scale

def sample_Gausses(mus, vars, rs, nb_samples_per_class = 500):
    """
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    num_classes = len(mus)
    out = []
    labels = []
    for c in range(num_classes):
      for i in range(nb_samples_per_class):
          out += [
              rs.normal(mus[c], vars[c])
          ]
      labels += [np.ones(nb_samples_per_class) * c]
    return np.stack(out, axis=0), np.stack(labels, axis=0).flatten()

class IrisDataset(Dataset):

  num_classes = 3

  def __init__(self,random=False,seed=28):

    self.X = np.load("./datasets/iris_data.npy").astype(np.float32)
    self.y = np.load("./datasets/iris_target.npy").astype(np.float32)

    self.input_size = self.X.shape[1]

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()
  
  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)

class GaussMixtureDataset(Dataset):
  # TODO check the random label vscode init that I progged on AMP
  # needs to have the data
  # for every seed, set itself into the randomly shuffeled state, with possibility to revert back
  # torch.randperm() is used to shuffle the labels

  means = [[2,-1], [-2,0], [0,2]]
  vars = [[0.5,0.5], [0.5,1], [0.5,0.5]]

  def __init__(self,random=False, four_classes=False, seed=28, num_samples = 300):
    self.num_samples = num_samples
    rs = np.random.RandomState(seed=42)

    self.num_classes = len(self.means)
    self.input_size = len(self.means[0])
    nb_samples_per_class = int(num_samples / self.num_classes)

    # to torch world
    means = self.means
    vars = self.vars

    X, y = sample_Gausses(self.means, self.vars, rs, nb_samples_per_class)

    self.X = X.astype(np.float32)
    self.y = y.astype(np.float32)

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)

class HierachicalGaussMixtureDataset(Dataset):
  # TODO check the random label vscode init that I progged on AMP
  # needs to have the data
  # for every seed, set itself into the randomly shuffeled state, with possibility to revert back
  # torch.randperm() is used to shuffle the labels

  means = [[2,-1],   [2,1], [-2,1], [-2,-1]]
  vars = [[0.5,0.55], [0.5,0.55], [0.5,0.55],[0.5,0.55]]

  def __init__(self,random=False, seed=34, num_samples = 300, num_classes = 4, random_type="uniform"):
    if num_classes not in [2,4]:
      raise ValueError("num_classes must be either 2 or 4")
    if random_type not in ["uniform", "within_hierachy"]:
      raise ValueError("random_type must be either 'uniform' or 'within_hierachy'")

    self.num_samples = num_samples
    rs = np.random.RandomState(seed=34)
    self.random_type = random_type

    self.num_classes = len(self.means)
    self.input_size = len(self.means[0])
    nb_samples_per_class = int(num_samples / self.num_classes)


    X, y = sample_Gausses(self.means, self.vars, rs, nb_samples_per_class)

    self.X = X.astype(np.float32)
    self.y = y.astype(np.float32)

    if num_classes == 2:
      self.y = np.where(self.y < 2, 0, 1)
      self.num_classes = 2

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    if self.random_type == "uniform":
      self.y_random = rs.permutation(self.y)
    elif self.random_type == "within_hierachy":
      self.y_random = self.y.copy()
      for c1,c2 in [[0,1],[2,3]]:
        idxs = np.where(np.logical_or(self.y == c1,self.y==c2))[0]
        self.y_random[idxs] = rs.permutation(self.y[idxs])
    else:
      raise ValueError("random_type must be either 'uniform' or 'within_hierachy'")
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)

class IntroDataset(Dataset):
  def __init__(self):
    self.num_classes = 2

    X = np.array([0.5,1.0,1.1,1.3,1.4,1.45]).reshape(-1,1)
    y = np.array([0,1,1,1,0,0])


    self.X = X.astype(np.float32)
    self.y = y.astype(np.float32)
    self.input_size = self.X.shape[1]
    self.num_samples = self.X.shape[0]




  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    y = self.y[idx]
    return self.X[idx], int(y)


class CircleDataset(Dataset):

  def __init__(self,random=False, seed=28, num_samples = 200, factor=0.5, noise=0.1):
    self.num_samples = num_samples
    rs = np.random.RandomState(seed=42)

    self.num_classes = 2

    X, y = make_circles(n_samples=num_samples, shuffle=True, noise=noise, random_state=seed, factor=factor)

    self.X = X.astype(np.float32)
    self.y = y.astype(np.float32)
    self.input_size = self.X.shape[1]

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)

class HalfMoonDataset(Dataset):

  def __init__(self,random=False, seed=28, num_samples = 200, noise=0.1):
    self.num_samples = num_samples
    rs = np.random.RandomState(seed=42)

    self.num_classes = 2

    X, y = make_moons(n_samples=num_samples, shuffle=True, noise=noise, random_state=seed)

    self.X = X.astype(np.float32)
    self.y = y.astype(np.float32)
    self.input_size = self.X.shape[1]

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)


class GaussCheckerboardLinearClose(Dataset):
  means = [[0.8,-3], [0.8,1], [-0.8,-1], [-0.8,3]]
  vars = [[0.3,0.1], [0.3,0.1], [0.3,0.1], [0.3,0.1]] 

  def __init__(self, random=False, seed=28, num_samples = 400):
    # to torch world
    means = self.means
    vars = self.vars
    
    self.num_samples = num_samples
    rs = np.random.RandomState(seed=41)

    self.num_classes = 2
    self.input_size = len(self.means[0])
    num_samples_per_class=100

    num_classblobs = 2
    nb_samples_per_class = int(num_samples / (self.num_classes * num_classblobs))
    
    out = []
    labels = []
    n=0
    for c in range(2):
        for num in range(num_classblobs):
            for i in range(nb_samples_per_class):
                out += [
                  rs.normal(means[n], vars[n])
              ]
            labels += [np.ones(nb_samples_per_class) * c]
            n+=1
    X = np.stack(out, axis=0)
    y = np.stack(labels, axis=0).flatten()


    self.X = X.astype(np.float32)
    self.y = y

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False
  
  def four_classes(self):
    fourth_class = np.intersect1d(np.argwhere(self.X[:,0] > 0.0), np.argwhere(self.X[:,1] > 0.5))
    self.y[fourth_class] = 3
    self.num_classes = 4

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)


class GaussCheckerboardNoisyClose(Dataset):
  means = [[0.5,-3], [0.5,1], [-0.5,-1], [-0.5,3]]
  vars = [[0.3,0.1], [0.3,0.1], [0.3,0.1], [0.3,0.1]] 

  def __init__(self, random=False, seed=28, num_samples = 400):
    # to torch world
    means = self.means
    vars = self.vars
    
    self.num_samples = num_samples
    rs = np.random.RandomState(seed=41)

    self.num_classes = 2
    self.input_size = len(self.means[0])
    num_samples_per_class=100

    num_classblobs = 2
    nb_samples_per_class = int(num_samples / (self.num_classes * num_classblobs))
    
    out = []
    labels = []
    n=0
    for c in range(2):
        for num in range(num_classblobs):
            for i in range(nb_samples_per_class):
                out += [
                  rs.normal(means[n], vars[n])
              ]
            labels += [np.ones(nb_samples_per_class) * c]
            n+=1
    X = np.stack(out, axis=0)
    y = np.stack(labels, axis=0).flatten()


    self.X = X.astype(np.float32)
    self.y = y

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    rs = np.random.RandomState(seed=self.seed)
    self.y_random = rs.permutation(self.y)
    self.act_random = True

  def true_labels(self):
    self.act_random = False
  
  def four_classes(self):
    fourth_class = np.intersect1d(np.argwhere(self.X[:,0] > 0.0), np.argwhere(self.X[:,1] > 0.5))
    self.y[fourth_class] = 3
    self.num_classes = 4

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], int(y)


class MySubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    loaded = None
    loaded_y = None

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def all(self):
      if self.loaded is None:
        if torch.is_tensor(self.dataset[0][0]):
          self.loaded = self.dataset[self.indices][0]
        else:
          self.loaded = torch.Tensor(np.array([self.dataset[idx][0] for idx in self.indices]))
      return self.loaded

    @property
    def labels(self):
      if self.loaded_y is None:
        if torch.is_tensor(self.dataset[0][1]):
          self.loaded_y = self.dataset[self.indices][1]
        else:
          self.loaded_y = torch.Tensor(np.array([self.dataset[idx][1] for idx in self.indices]))
      return self.loaded_y.long()


def random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [MySubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class MNIST2DDataset(Dataset):

  def __init__(self, num_samples_per_cls=200, class_list=[0,1,2], train = False, random=False, seed=28):
    mnist_data = datasets.MNIST(root=DATASET_DIR, train=train, transform=ToTensor(), download=True)

    x_idx = torch.tensor([])
    for i in class_list:
      x_idx = torch.cat((x_idx, (mnist_data.targets == i).nonzero(as_tuple=True)[0][:num_samples_per_cls]))
    x_idx = x_idx.long()

    self.X = mnist_data.data.reshape(mnist_data.data.shape[0], 1, mnist_data.data.shape[1], mnist_data.data.shape[2])
    self.X = self.X[x_idx]
    self.y = mnist_data.targets[x_idx]
    self.num_classes = len(class_list)

    #convert labels
    y_idx_list = [torch.where(self.y==i)[0] for i in class_list]
    for y_idx in range(len(y_idx_list)):
        self.y[y_idx_list[y_idx]] = y_idx

    self.input_size = self.X.shape[1]

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    idx = torch.randperm(self.y.shape[0], generator=torch.Generator().manual_seed(self.seed))
    self.y_random = self.y[idx]
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], y.to(torch.long) #int(y)
  
class CIFAR10(Dataset):

  def __init__(self, num_samples_per_cls = 200, class_list = [0,1,2], train = False, random = False, seed = 28):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_data = datasets.CIFAR10(root = DATASET_DIR, train = train, transform = transform, download = True)
    x_idx = torch.tensor([])
    for i in class_list:
      x_idx = torch.cat((x_idx, (torch.tensor(cifar_data.targets) == i).nonzero(as_tuple=True)[0][:num_samples_per_cls]))
    x_idx = x_idx.long()

    self.X = np.swapaxes(cifar_data.data,1,3) #.reshape(cifar_data.data.shape[0], 1, cifar_data.data.shape[1], cifar_data.data.shape[2])
    self.X = self.X[x_idx]
    self.y = torch.tensor(cifar_data.targets)[x_idx]
    self.num_classes = len(class_list)

    #convert labels
    y_idx_list = [torch.where(self.y==i)[0] for i in class_list]
    for y_idx in range(len(y_idx_list)):
        self.y[y_idx_list[y_idx]] = y_idx

    self.input_size = self.X.shape[1]

    self.act_random = False

    self.seed = seed

    if random:
      self.random_labels()

  def random_labels(self):
    idx = torch.randperm(self.y.shape[0], generator=torch.Generator().manual_seed(self.seed))
    self.y_random = self.y[idx]
    self.act_random = True

  def true_labels(self):
    self.act_random = False

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    if self.act_random:
      y = self.y_random[idx]
    else:
      y = self.y[idx]
    return self.X[idx], y.to(torch.long) #int(y)
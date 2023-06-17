import numpy as np
import torch.nn as nn
from src.hessian.grads import compute_reinforcing_gradients
import matplotlib.pyplot as plt

def generalization_measure(overlaps,zero = 1e-6):
    overlap_mean = np.max(np.abs(overlaps), axis=1) # can use this for plotting if needed
    overlap_spread = np.copy(overlap_mean)
    overlap_spread[overlap_spread < zero] = 0
    overlap_spread[overlap_spread > 0] = 1
    ratio_eigvec_spread = np.sum(overlap_spread)/overlap_spread.shape[0]
    return ratio_eigvec_spread

def compute_threshold(model, X, rand_vec_dim, k=5, criterion = nn.CrossEntropyLoss(), mnist=False, cos_sim=True):
    rs = np.random.RandomState(42)
    vectors = [v / np.linalg.norm(v) for v in [rs.randn(rand_vec_dim) for _ in range(k)]]
    overlaps = compute_reinforcing_gradients(model, X, vectors, criterion = criterion, mnist=mnist, cos_sim=cos_sim)
    max_overlap = np.max(np.abs(overlaps), axis=1)
    threshold = np.min(max_overlap), np.mean(max_overlap)
    print('min of ', max_overlap)
    return threshold

def get_gen_measures(heigenvectors, heigenvalues, train_data, model, criterion, mnist=False, epsilon=1e-6):
    vectors = [heigenvectors[:, which_heigenvector]
               for which_heigenvector in range(heigenvectors.shape[1])]
    overlaps = compute_reinforcing_gradients(
        model, train_data.all, vectors, criterion, mnist)
    gen = generalization_measure(overlaps, zero=epsilon)
    tr = sum(heigenvalues)
    l1 = heigenvalues[-1]
    return (gen, tr, l1)

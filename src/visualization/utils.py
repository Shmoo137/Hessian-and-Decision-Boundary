import numpy as np
import scipy as sp

"""
Function to help generating plots.
"""

def generate_grid_data(min_x, max_x, min_y, max_y, dim_x, dim_y):
    x1 = np.linspace(min_x, max_x, dim_x)
    x2 = np.linspace(min_y, max_y, dim_y)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2, np.vstack([X1.ravel(), X2.ravel()]).T


def show_training_points(ax, train_data,highlight=None):
    markers = ['s', 'o', 'd', '+', 'x']
    label_classes = np.unique(train_data.labels)
    if highlight is not None:
        label_classes = highlight
    for i, c in enumerate(label_classes):
        mask = train_data.labels == c
        ax.scatter(train_data.all[:, 0][mask], train_data.all[:, 1][mask],
                   marker=markers[i], color='black', edgecolors='black', s=3)

    # keep in case we want to add legend
    # legend_elements = [Line2D([0], [0], marker=marker, color='black', label=label, markersize=4) for label,marker in enumerate(markers[:label_classes])]


def show_decision_boundaries(ax, X1, X2, Y, as_scatter=False):
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
        ax.scatter(X1[bn_img], X2[bn_img], c='black', s=3)
    else:
        ax.pcolor(X1, X2, bn_img, cmap='Greys')


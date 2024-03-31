import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

c_vibrant = {'blue': '#0077BB',
             'cyan': '#33BBEE',
             'teal': '#009988',
             'orange': '#EE7733',
             'red': '#CC3311',
             'magenta': '#EE3377',
             'grey': '#BBBBBB'}

c_vibrant_array = ['#0077BB',
                   '#33BBEE',
                   '#009988',
                   '#EE7733',
                   '#CC3311',
                   '#EE3377',
                   '#BBBBBB',
                   '#AAAA00',
                   '#DDDDDD',
                   '#332288',
                   '#DDCC77',
                   '#44BB99'
                   ]

c_light = {'light blue': '#77AADD',
           'light_cyan': '#EE8866',
           'min': '#EEDD88',
           'pear': '#FFAABB',
           'olive': '#99DDFF',
           'light yellow': '#44BB99',
           'orange': '#BBCC33',
           'pink': '#AAAA00',
           'pale grey': '#DDDDDD'}

c_muted = {'indigo': '#332288',
           'cyan': '#88CCEE',
           'teal': '#44AA99',
           'green': '#117733',
           'olive': '#999933',
           'sand': '#DDCC77',
           'rose': '#CC6677',
           'wine': '#883355',
           'purple': '#AA4499',
           'grey': '#DDDDDD'}

c_muted_array = ['#332288',
                 '#88CCEE',
                 '#44AA99',
                 '#117733',
                 '#999933',
                 '#DDCC77',
                 '#CC6677',
                 '#883355',
                 '#AA4499',
                 '#DDDDDD']

zero = 1e-6

FIGWIDTH_COLUMN = 7
FIGWIDTH_FULL = 20

CLASS_COLORS = [c_vibrant['cyan'], c_vibrant['magenta'],
                c_vibrant['teal'], c_light['pear']]

INIT_COLORS = {
    'normal': c_muted['indigo'],
    'adversarial': c_muted['teal'],
    'large norm': c_muted['purple'],
    'sharper': c_muted['rose']
}

n_bin = 100

cmap_YWPu = LinearSegmentedColormap.from_list('YWPu',
                                              [(255/255, 182/255, 78/255), (1, 1,
                                                                            1), (175/255, 179/255, 239/255)],
                                              N=n_bin)  # yellow-white-purple

cmap_RWC = LinearSegmentedColormap.from_list('RWC',
                                             [(255/255, 72/255, 88/255), (1, 1,
                                                                          1), (0/255, 204/255, 192/255)],
                                             N=n_bin)  # red-white-cyan


cmap_PiWY = LinearSegmentedColormap.from_list('PiWY',
                                              [(242/255, 5/255, 113/255), (1, 1,
                                                                           1), (235/255, 196/255, 5/255)],
                                              N=n_bin)  # pink-white-yellow

cmap_PiWG = LinearSegmentedColormap.from_list('PiWG',
                                              [(242/255, 5/255, 113/255), (1, 1,
                                                                           1), (87/255, 187/255, 66/255)],
                                              N=n_bin)  # pink-white-yellow

cmap_YWPu_dark = LinearSegmentedColormap.from_list('PiWY',
                                                   [(242/255, 172/255, 45/255), (1, 1, 1), (143/255, 146/255, 227/255)], N=n_bin)
CMAP_OVERLAP = cmap_YWPu_dark  # 'PiYG'

CMAP_MNIST = sns.color_palette(n_colors=10)


def init():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    # Matplotlib settings
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.formatter.limits'] = (-3, 3)
    mpl.rcParams['axes.formatter.use_mathtext'] = True

    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['legend.labelspacing'] = 0.3
    mpl.rcParams['legend.borderpad'] = 0.2
    mpl.rcParams['legend.handlelength'] = 1

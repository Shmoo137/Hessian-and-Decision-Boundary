from math import sqrt, copysign
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

SPINE_COLOR = 'black'

# COLORS from # https://personal.sron.nl/~pault/

c_high_contrast = ['#DDAA33',
                 '#BB5566',
                 '#004488',
                 '#000000']
c_vibrant = {'blue': '#0077BB',
           'cyan': '#33BBEE',
           'teal': '#009988',
           'orange': '#EE7733',
           'red': '#CC3311',
           'magenta': '#EE3377',
           'grey': '#BBBBBB'}

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

c_medium_contrast = {'light-yellow': '#EECC66',
                   'light-red': '#EE99AA',
                   'light-blue': '#6699CC',
                   'dark-yellow': '#997700',
                   'dark-red': '#994455',
                   'dark-blue': '#004488',
                   'black': '#000000'}

c_bright = {'blue': '#4477AA',
          'cyan': '#66CCEE',
          'green': '#228833',
          'yellow': '#CCBB44',
          'red': '#EE6677',
          'purple': '#AA3377',
          'grey': '#BBBBBB'}

c_light = {'light blue': '#77AADD',
'light_cyan': '#EE8866', 
'min':'#EEDD88', 
'pear':'#FFAABB', 
'olive':'#99DDFF',
 'light yellow':'#44BB99',
  'orange':'#BBCC33', 
  'pink':'#AAAA00',
  'pale grey': '#DDDDDD'}


# from https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html

def small_latexify():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Lato'], size=12)


def latexify(fig_width=None, fig_height=None, columns=1,color_scheme=c_vibrant):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': r'\usepackage{gensymb}',
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)
    plt.rc('axes', prop_cycle=(cycler('color', color_scheme.values())))


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')

    #for axis in [ax.xaxis, ax.yaxis]:
    #    axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

## Common marker/line settings

def empty_O_marker(color,width=1):
    return {
        'edgecolor': color,
        'facecolor': 'none',
        'marker': 'o',
        'linewidth': width
    }

def shiftedValueForMesh(value, percentage):
    if copysign(1, value) > 0:
        shifted_value =  value * (1 + percentage)
    else:
        shifted_value =  value * (1 - percentage)

    return shifted_value

def buildMeshgrid(arrayA, arrayB, percentage = 0.1, dim = 100):
    # define mesh that depends on the input data
    min_x, max_x = shiftedValueForMesh(np.amin(arrayA), - percentage), shiftedValueForMesh(np.amax(arrayA), percentage) 
    min_y, max_y = shiftedValueForMesh(np.amin(arrayB), - percentage), shiftedValueForMesh(np.amax(arrayB), percentage) 
    #print("Min and max element of array A: ", np.amin(arrayA), np.amax(arrayA), " min and max values for mesh: ", min_x, max_x)
    #print("Min and max element of array B: ", np.amin(arrayB), np.amax(arrayB), " min and max values for mesh: ", min_y, max_y)
    dim_x = dim_y = dim
    x = np.linspace(min_x, max_x, dim_x)
    y = np.linspace(min_y, max_y, dim_y)
    X, Y = np.meshgrid(x,y)
    
    return x, y, X, Y

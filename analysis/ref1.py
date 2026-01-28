import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
import matplotlib.ticker as mtick
import pandas as pd
import pickle
import itertools

config = {
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
mpl.rc('pdf', fonttype=42)
FONTSIZE = 20
ALLWIDTH = 1.5
Marker = ['o', 'v', '8', 's', 'p', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X']
HATCH = ['+', 'x', '/', 'o', '|', '\\', '-', 'O', '.', '*']
Line_Style = ['-', '--', '-.', ':']
COLORS = sns.color_palette("Paired")
rcParams.update(config)


def bar1_plot():
    width = 0.2

    dataset = 'WF'
    #dataset = 'iot'
    #dataset = 'ISCX'

    # memory 
    if dataset == 'WF':
        y = np.array([[98.795, 98.645, 99.398], [8.13, 10.59, 8.94], [14.42, 17.14, 9.84], [52.751, 53.290, 53.585], [18.98, 9.89, 14.01], [3.765, 2.355, 3.711], [3.012, 1.958, 1.657]])
    elif dataset == 'iot':
        y = np.array([[90.200, 83.301, 86.525], [47.96, 40.12, 41.76], [44.50, 48.72, 50.85], [90.200, 83.301, 86.525], [29.85, 32.62, 32.30], [25.145, 23.404, 28.756], [23.727, 18.762, 18.569]])
    elif dataset == 'ISCX':
        y = np.array([[78.800, 79.800, 72.600], [73.49, 67.43, 38.62], [19.20, 16.20, 24.60], [76.300, 77.269, 70.297], [16.66, 16.66, 20.95], [22.400, 22.800, 8.200], [30.600, 24.400, 13.000]])

    x_labels = ['No Defense', 'Minipatch', 'BLANKET', 'SPINE', 'WTF-PAD', 'Ditto', 'Securitas']
    #x_labels = ['Minipatch', 'BLANKET', 'WTF-PAD', 'Ditto', 'SPINE', 'Securitas']
    if dataset == 'WF':
        #legend_labels = ['DF(98.80%)', 'AWF(98.65%)', 'Var-CNN(99.40%)']
        legend_labels = ['DF', 'AWF', 'Var-CNN']
    elif dataset == 'iot':
        #legend_labels = ['ANN(90.20%)', 'LSTM(83.30%)', 'BILSTM(86.53%)']
        legend_labels = ['ANN', 'LSTM', 'BILSTM']
    elif dataset == 'ISCX':
        #legend_labels = ['App-Net(78.80%)', 'FS-Net(79.80%)', 'Transformer(72.60%)']
        legend_labels = ['App-Net', 'FS-Net', 'Transformer']
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(6.5, 3.6)) # 6.65, 3.65 # 4.1, 2.8 * 510 / 527
    ax.bar(x[0:] - 1.0 * width, y[0:, 0] , width=width, label=legend_labels[0], color='white',
           ec=COLORS[1], hatch=HATCH[1] * 2, linewidth=ALLWIDTH)
    ax.bar(x[0:] - 0.0 * width, y[0:, 1] , width=width, label=legend_labels[1], color='white',
           ec=COLORS[9], hatch=HATCH[1] * 2, linewidth=ALLWIDTH)
    ax.bar(x[0:] + 1.0 * width, y[0:, 2] , width=width, label=legend_labels[2], color='white',
           ec=COLORS[3], hatch=HATCH[5] * 2, linewidth=ALLWIDTH)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', rotation_mode='anchor')
    #ax.set_xticklabels(x_labels, rotation=18)
    ax.set_ylabel('Attack Accuracy(%)', fontsize=FONTSIZE)

    if dataset == 'WF':
        plt.ylim(0, 110)
        y_major_locator = MultipleLocator(25)
    elif dataset == 'iot':
        plt.ylim(0, 110)
        y_major_locator = MultipleLocator(25)
    elif dataset == 'ISCX':
        plt.ylim(0, 110)
        y_major_locator = MultipleLocator(25)
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    ax.tick_params(labelsize=FONTSIZE)
    plt.tick_params(axis='both', which='both', length=0)
    ax.grid(linestyle=':', axis='y')
    if dataset == 'WF':
        fig.legend(fontsize=FONTSIZE, loc='upper left', ncol=1, handleheight=0.7,
                handlelength=1.2, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.67, 0.94)))
    elif dataset == 'iot':
        fig.legend(fontsize=FONTSIZE, loc='upper left', ncol=1, handleheight=0.7,
                handlelength=1.2, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.68, 0.95)))
    elif dataset == 'ISCX':
        fig.legend(fontsize=FONTSIZE, loc='upper left', ncol=1, handleheight=0.7,
                handlelength=1.2, handletextpad=0.2, columnspacing=1, frameon=True, bbox_to_anchor=((0.63, 0.959)))
    plt.tight_layout()

    pp = PdfPages('%s.pdf' % (dataset))
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    plt.show()

if __name__=="__main__":
    bar1_plot()
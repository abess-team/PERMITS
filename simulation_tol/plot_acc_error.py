import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import objects as so
from matplotlib.ticker import NullFormatter
import warnings
warnings.filterwarnings('ignore')


markers = ['D', '>', '>', '>', 'o', 's', 'X']
dashes = [(2, 2), (1, 1), (1, 1), (1, 1), (3, 3), (3, 3), (5, 0)]
palette = ['deepskyblue', 'deeppink', 'darkorange', 'darkseagreen', 'darkblue', 'purple', 'forestgreen']

def plot(df, y, ylim, row, col, figsize=None):
    row_list = np.sort(df[row].unique())
    col_list = np.sort(df[col].unique())
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for (i, j), ax in np.ndenumerate(axes):
        row_value = row_list[i]
        col_value = col_list[j]
        sns.lineplot(
            data=df[(df[row]==row_value) & (df[col]==col_value)], 
            hue='model', 
            x='n', 
            y=y, 
            estimator='mean', 
            errorbar=None, 
            style='model', 
            markers=markers,
            dashes=dashes,
            palette=palette, 
            linewidth=0.5,
            ax=ax,
        )
        
        ax.get_legend().remove()
        ax.set(ylabel=None)
        ax.set(xlabel=None)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
        ax.set_xticks(np.arange(100, 601, 100))
        if i < 2:
            ax.axes.get_xaxis().set_major_formatter(NullFormatter())
        if j > 0:
            ax.axes.get_yaxis().set_major_formatter(NullFormatter())
        else:
            # axes[i, j].yaxis.tick_right()
            pass
        ax.set_ylim(*ylim[i])
        ax.set_xlim(80, 620)
        ax.grid(True)
    
    axes[2, 1].set_xlabel('Sample size')
    if y == 'accuracy':
        axes[1, 0].set_ylabel('Accuracy')
    elif y == 'err':
        axes[1, 0].set_ylabel('Error')
    else:
        axes[1, 0].set_ylabel(y)

    for i in range(3):
        axes[0, i].set_title(f'Exp({col_list[i]})')
        axes[i, 2].set_ylabel(f'{row_list[i]}')
        axes[i, 2].yaxis.set_label_position('right')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.06), ncol=6
    )
    fig.tight_layout()
    return fig


df = pd.read_csv('./tol_acc_error.csv')
df = df[~df['model'].isin(['SCOPE'])]
df = df.replace(
    'IHT', 'IHT*').replace(
        'PERMITS-0.001', 'PERMITS (e-3)').replace(
            'PERMITS-0.0001', 'PERMITS (e-4)').replace(
            'PERMITS-1e-05', 'PERMITS (e-5)')
row, col = 'snr', 'corr'

# 1. Accuracy
fig_accuracy = plot(df=df, y='accuracy', ylim=[(0, 1.05)]*3, 
                             row=row, col=col, figsize=(10, 6))
fig_accuracy.savefig('./tol_accuracy.pdf', bbox_inches='tight', dpi=500) 

# 2. Error
fig_error = plot(df=df, y='err', ylim=[(0, 0.5), (0, 0.5), (0, 0.3)], 
                          row=row, col=col, figsize=(10, 6))
fig_error.savefig('./tol_error.pdf', bbox_inches='tight', dpi=500)

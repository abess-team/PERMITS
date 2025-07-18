import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import objects as so
from matplotlib.ticker import NullFormatter
import warnings
warnings.filterwarnings('ignore')


markers = ['D', '>', 'o', 's', 'X']
dashes = [(2, 2), (1, 1), (3, 3), (3, 3), (5, 0)]
palette = ['deepskyblue', 'deeppink', 'darkblue', 'purple', 'forestgreen']

def plot(df, x, y, xlim, ylim, xlabel, figsize=None):
    corr_list = np.sort(df['corr'].unique())
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i in range(len(axes)):
        ax = axes[i]
        corr = corr_list[i]
        sns.lineplot(
            data=df[df['corr']==corr], 
            hue='model', 
            x=x, 
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
        ax.set_yticks(np.arange(0, 3.1, 0.5))
        # ax.set_xticks(np.arange(0, 1001, 200))
        ax.set_title(f'Exp({corr_list[i]})')
        if i > 0:
            ax.axes.get_yaxis().set_major_formatter(NullFormatter())
        ax.set_xlim(*xlim[i])
        ax.set_ylim(*ylim[i])
        ax.grid(True)
    
    axes[1].set_xlabel(xlabel)
    axes[0].set_ylabel('Time')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.06), ncol=5
    )
    fig.tight_layout()
    return fig


df_p = pd.read_csv('./time_dimension.csv')
# df_p = df_p[df_p['corr'] == 0.5]
df_p = df_p.replace('IHT', 'IHT*').replace('SCOPE', 'SCOPE*').replace('LIMITS', 'PERMITS')
df_p = df_p[df_p['model'] != 'SCOPE*']
fig_time_p = plot(df=df_p, x='p', y='time', 
                  xlim=[(80, 1020)]*3, ylim=[(-0.1, 2)]*3, xlabel='Dimension',
                  figsize=(10, 3))
# fig_time_p.savefig('./time_dimension.pdf', bbox_inches='tight', dpi=500) 